#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ContextManager, Iterator, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf
from gguf import *
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, LlamaTokenizer


# check for any of the given keys in the dictionary and return the value of the first key found
def get_key_opts(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    print(f"Could not find any of {keys}")
    sys.exit()


###### MODEL DEFINITIONS ######

class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class CogVLM:
    language_tensor_name_mapping_layers = (
        # DFF model
        ("model.embed_tokens.weight", "token_embd.weight", False, None),
        ("model.norm.weight", "output_norm.weight", False, None),
        ("lm_head.weight", "output.weight", False, None),
        ("model.layers.{}.input_layernorm.weight", "blk.{}.attn_norm.weight", False, None),
        ("model.layers.{}.self_attn.language_expert_query_key_value.weight", "blk.{}.attn_{}.0.weight", True,
         ["q", "k", "v"]),
        ("model.layers.{}.self_attn.language_expert_dense.weight", "blk.{}.attn_output.0.weight", False, None),
        ("model.layers.{}.self_attn.vision_expert_query_key_value.weight", "blk.{}.attn_{}.1.weight", True,
         ["q", "k", "v"]),
        ("model.layers.{}.self_attn.vision_expert_dense.weight", "blk.{}.attn_output.1.weight", False, None),
        ("model.layers.{}.mlp.language_mlp.down_proj.weight", "blk.{}.ffn_down.0.weight", False, None),
        ("model.layers.{}.mlp.language_mlp.gate_proj.weight", "blk.{}.ffn_gate.0.weight", False, None),
        ("model.layers.{}.mlp.language_mlp.up_proj.weight", "blk.{}.ffn_up.0.weight", False, None),
        ("model.layers.{}.mlp.vision_mlp.down_proj.weight", "blk.{}.ffn_down.1.weight", False, None),
        ("model.layers.{}.mlp.vision_mlp.gate_proj.weight", "blk.{}.ffn_gate.1.weight", False, None),
        ("model.layers.{}.mlp.vision_mlp.up_proj.weight", "blk.{}.ffn_up.1.weight", False, None),
        ("model.layers.{}.post_attention_layernorm.weight", "blk.{}.ffn_norm.weight", False, None),
    )

    def __init__(self, dir_model: Path, ftype: int, fname_out: Path, is_big_endian: bool):
        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.is_safetensors = self._is_model_safetensors()
        self.num_parts = CogVLM.count_model_parts(self.dir_model, ".safetensors" if self.is_safetensors else ".bin")
        self.part_names = self._get_part_names()
        self.hparams = CogVLM.load_hparams(self.dir_model)
        self.model_arch = self._get_model_architecture()
        self.gguf_writer = gguf.GGUFWriter(fname_out, arch=self.model_arch, endianess=self.endianess,
                                           use_temp_file=False)

        self.language_extra_mapping_args = {}
        self.ftype_str = ["f32", "f16"]

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for part_name in self.part_names:
            print(f"gguf: loading model part '{part_name}'")
            ctx: ContextManager[Any]
            if self.is_safetensors:
                from safetensors import safe_open
                ctx = cast(ContextManager[Any], safe_open(self.dir_model / part_name, framework="pt", device="cpu"))
            else:
                ctx = contextlib.nullcontext(
                    torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True))

            with ctx as model_part:
                for name in model_part.keys():
                    data = model_part.get_tensor(name) if self.is_safetensors else model_part[name]
                    yield name, data

    def set_gguf_parameters(self):
        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_block_count(self.hparams.get(
            "n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")),
        ))
        if (n_ctx := self.hparams.get("max_position_embeddings")) is not None:
            self.gguf_writer.add_context_length(n_ctx)
        if (n_embd := self.hparams.get("hidden_size")) is not None:
            self.gguf_writer.add_embedding_length(n_embd)
        if (n_ff := self.hparams.get("intermediate_size")) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
        if (n_head := self.hparams.get("num_attention_heads")) is not None:
            self.gguf_writer.add_head_count(n_head)
        if (n_head_kv := self.hparams.get("num_key_value_heads")) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)

        if (n_rms_eps := self.hparams.get("rms_norm_eps")) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(n_rms_eps)

        self.gguf_writer.add_uint32(f"{self.model_arch}.expert_count", 2)
        self.gguf_writer.add_uint32(f"{self.model_arch}.expert_used_count", 2)

        self.gguf_writer.add_parallel_residual(self.hparams.get("use_parallel_residual", True))

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = {}
        for l in range(block_count):
            for layer_spec in self.language_tensor_name_mapping_layers:
                py_name = layer_spec[0].format(l)
                if layer_spec[2]:
                    llama_names = [layer_spec[1].format(l, a) for a in layer_spec[3]]
                    self.language_extra_mapping_args[py_name] = llama_names
                    continue

                llama_name = layer_spec[1].format(l)
                tensor_map[py_name] = llama_name

        for name, data_torch in self.get_tensors():
            mapped_name = tensor_map.get(name, None)
            if mapped_name is None:
                language_extra_args = self.language_extra_mapping_args.get(name, None)
                if language_extra_args:
                    if "weight" in name:
                        data = data_torch.reshape(3, self.hparams["hidden_size"], self.hparams["hidden_size"]).float().squeeze().numpy()
                    elif "bias" in name:
                        data = data_torch.reshape(3, self.hparams["hidden_size"]).float().squeeze().numpy()
                    for idx, e in enumerate(language_extra_args):
                        self.gguf_writer.add_tensor(e, data[idx, ...])
                        print(f"tensor {e} is always saved in f16")
                    continue
                else:
                    continue

            name = mapped_name

            data = data_torch.float().squeeze().numpy()

            n_dims = len(data.shape)

            # ftype == 0 -> float32, ftype == 1 -> float16
            ftype_cur = 0
            if n_dims == 4:
                print(f"tensor {name} is always saved in f16")
                data = data.astype(np.float16)
                ftype_cur = 1
            elif self.ftype == 1:
                if name[-7:] == ".weight" and n_dims == 2:
                    print("  Converting to float16")
                    data = data.astype(np.float16)
                    ftype_cur = 1
                else:
                    print("  Converting to float32")
                    data = data.astype(np.float32)
                    ftype_cur = 0
            else:
                if data.dtype != np.float32:
                    print("  Converting to float32")
                    data = data.astype(np.float32)
                    ftype_cur = 0

            print(f"{name} - {self.ftype_str[ftype_cur]} - shape = {data.shape}")
            self.gguf_writer.add_tensor(name, data)

    def write(self):
        self.write_tensors()
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file()
        self.gguf_writer.close()

    def write_vocab(self):
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    @staticmethod
    def count_model_parts(dir_model: Path, prefix: str) -> int:
        num_parts = 0
        for filename in os.listdir(dir_model):
            if filename.endswith(prefix):
                num_parts += 1

        return num_parts

    @staticmethod
    def load_hparams(dir_model):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def from_model_architecture(model_architecture):
        if model_architecture == "CogVLMForCausalLM":
            return CogVLM
        raise NotImplementedError(f'Script only supports CogVLM conversion!')

    def _is_model_safetensors(self) -> bool:
        return CogVLM.count_model_parts(self.dir_model, ".safetensors") > 0

    def _get_part_names(self):
        if self.is_safetensors:
            if self.num_parts == 1:  # there's only one .safetensors file
                return ("model.safetensors",)
            return (f"model-{n:05}-of-{self.num_parts:05}.safetensors" for n in range(1, self.num_parts + 1))

        if self.num_parts == 1:  # there's only one .bin file
            return ("pytorch_model.bin",)
        return (f"pytorch_model-{n:05}-of-{self.num_parts:05}.bin" for n in range(1, self.num_parts + 1))

    def _get_model_architecture(self) -> gguf.MODEL_ARCH:
        arch = self.hparams["architectures"][0]
        if arch == "CogVLMForCausalLM":
            return "deepfeaturefusion"

        raise NotImplementedError(f'Architecture "{arch}" not supported!')

    def _set_vocab_sentencepiece(self):
        from sentencepiece import SentencePieceProcessor
        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        tokenizer_path = Path(tokenizer.vocab_file).parent
        tokenizer_path = tokenizer_path / 'tokenizer.model'

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        if not tokenizer_path.is_file():
            print(f'Error: Missing {tokenizer_path}', file=sys.stderr)
            sys.exit(1)

        tokenizer = SentencePieceProcessor(str(tokenizer_path))
        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        for token_id in range(vocab_size):
            piece = tokenizer.id_to_piece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.get_score(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.is_unknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.is_control(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.is_unused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.is_byte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    tokens.append(key.encode("utf-8"))
                    scores.append(-1000.0)
                    toktypes.append(SentencePieceTokenTypes.USER_DEFINED)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)


###### CONVERSION LOGIC ######


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--awq-path", type=Path, default=None,
        help="Path to scale awq cache file")
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16"], default="f16",
        help="output format - use f32 for float32, f16 for float16",
    )
    parser.add_argument("--bigendian", action="store_true", help="model is executed on big endian machine")
    parser.add_argument(
        "model", type=Path,
        help="directory containing model file",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dir_model = args.model

    if args.awq_path:
        sys.path.insert(1, str(Path(__file__).parent / 'awq-py'))
        from awq.apply_awq import add_scale_weights  # type: ignore[import-not-found]
        tmp_model_path = args.model / "weighted_model"
        dir_model = tmp_model_path
        if tmp_model_path.is_dir():
            print(f"{tmp_model_path} exists as a weighted model.")
        else:
            tmp_model_path.mkdir(parents=True, exist_ok=True)
            print("Saving new weighted model ...")
            add_scale_weights(str(args.model), str(args.awq_path), str(tmp_model_path))
            print(f"Saved weighted model at {tmp_model_path}.")

    if not dir_model.is_dir():
        print(f'Error: {args.model} is not a directory', file=sys.stderr)
        sys.exit(1)

    ftype_map = {
        "f32": gguf.GGMLQuantizationType.F32,
        "f16": gguf.GGMLQuantizationType.F16,
    }

    if args.outfile is not None:
        fname_out = args.outfile
    else:
        # output in the same directory as the model by default
        fname_out = dir_model / f'ggml-model-{args.outtype}.gguf'

    print(f"Loading model: {dir_model.name}")

    hparams = CogVLM.load_hparams(dir_model)

    with torch.inference_mode():
        model_class = CogVLM.from_model_architecture(hparams["architectures"][0])
        model_instance = model_class(dir_model, ftype_map[args.outtype], fname_out, args.bigendian)

        print("Set model parameters")
        model_instance.set_gguf_parameters()

        print("Set model tokenizer")
        model_instance.set_vocab()

        if args.vocab_only:
            print(f"Exporting model vocab to '{fname_out}'")
            model_instance.write_vocab()
        else:
            print(f"Exporting model to '{fname_out}'")
            model_instance.write()

        print(f"Model successfully exported to '{fname_out}'")


if __name__ == '__main__':
    main()
