import argparse

import torch
from gguf import *
from transformers import AutoModelForCausalLM, AutoConfig

TEXT = "clip.text"
VISION = "clip.vision"


class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6

vision_tensor_name_mapping_layers = (
    ("model.vision.transformer.layers.{}.attention.dense.bias", "v.blk.{}.attn_out.bias", False, None),
    ("model.vision.transformer.layers.{}.attention.dense.weight", "v.blk.{}.attn_out.weight", False, None),
    ("model.vision.transformer.layers.{}.attention.query_key_value.weight", "v.blk.{}.attn_{}.weight", True,
     ["q", "k", "v"]),
    ("model.vision.transformer.layers.{}.attention.query_key_value.bias", "v.blk.{}.attn_{}.bias", True,
     ["q", "k", "v"]),
    ("model.vision.transformer.layers.{}.input_layernorm.bias", "v.blk.{}.ln1.bias", False, None),
    ("model.vision.transformer.layers.{}.input_layernorm.weight", "v.blk.{}.ln1.weight", False, None),
    ("model.vision.transformer.layers.{}.post_attention_layernorm.bias", "v.blk.{}.ln2.bias", False, None),
    ("model.vision.transformer.layers.{}.post_attention_layernorm.weight", "v.blk.{}.ln2.weight", False, None),
    ("model.vision.transformer.layers.{}.mlp.fc1.bias", "v.blk.{}.ffn_down.bias", False, None),
    ("model.vision.transformer.layers.{}.mlp.fc1.weight", "v.blk.{}.ffn_down.weight", False, None),
    ("model.vision.transformer.layers.{}.mlp.fc2.bias", "v.blk.{}.ffn_up.bias", False, None),
    ("model.vision.transformer.layers.{}.mlp.fc2.weight", "v.blk.{}.ffn_up.weight", False, None),
    ("model.vision.linear_proj.linear_proj.weight", "v.linear_proj.weight", False, None),
    ("model.vision.linear_proj.dense_h_to_4h.weight", "v.dense_h_to_4h.weight", False, None),
    ("model.vision.linear_proj.dense_4h_to_h.weight", "v.dense_4h_to_h.weight", False, None),
    ("model.vision.linear_proj.gate_proj.weight", "v.gate_proj.weight", False, None),
    ("model.vision.linear_proj.norm1.weight", "v.norm1.weight", False, None),
    ("model.vision.linear_proj.norm1.bias", "v.norm1.bias", False, None),

)

vision_extra_mapping_args = {}
language_extra_mapping_args = {}

tensor_name_mapping = {
    "model.vision.patch_embedding.proj.weight": "v.patch_embd.weight",
    "model.vision.patch_embedding.proj.bias": "v.patch_embd.bias",
    "model.vision.patch_embedding.position_embedding.weight": "v.position_embd.weight",
    "model.vision.patch_embedding.cls_embedding": "v.class_embd",
    "model.vision.boi": "v.boi",
    "model.vision.eoi": "v.eoi"
}


def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def parse():
    ap = argparse.ArgumentParser(prog="convert_hf_to_gguf.py")
    ap.add_argument("-m", "--model-dir", help="Path to model directory cloned from HF Hub", required=True)
    # ap.add_argument("--use-f32", action="store_true", default=False, help="Use f32 instead of f16")
    # ap.add_argument("--text-only", action="store_true", required=False,
    #                help="Save a text-only model. It can't be used to encode images")
    # ap.add_argument("--vision-only", action="store_true", required=False,
    #                help="Save a vision-only model. It can't be used to encode texts")
    # ap.add_argument("--clip_model_is_vision", action="store_true", required=False,
    #                help="The clip model is a pure vision model (ShareGPT4V vision extract for example)")
    # ap.add_argument("--llava-projector", help="Path to llava.projector file. If specified, save an image encoder for LLaVA models.")
    # ap.add_argument("--image-mean", nargs=3, type=float, required=False, help="Override image mean values")
    # ap.add_argument("--image-std", nargs=3, type=float, required=False, help="Override image std values")
    ap.add_argument("-o", "--output-dir", help="Directory to save GGUF files. Default is the original model directory",
                    default=None)
    # Example --image_mean 0.48145466 0.4578275 0.40821073 --image_std 0.26862954 0.26130258 0.27577711

    ap.add_argument('--image_mean', type=float, nargs='+',
                    help='Mean of the images for normalization (overrides processor) ', default=None)
    ap.add_argument('--image_std', type=float, nargs='+',
                    help='Standard deviation of the images for normalization (overrides processor)', default=None)

    # with proper
    args = ap.parse_args()

    return args


if __name__ == "__main__":
    args = parse()

    # output in the same directory as the model if output_dir is None
    dir_model = args.model_dir

    with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        v_hparams = config["vision_config"]

    # possible data types
    #   ftype == 0 -> float32
    #   ftype == 1 -> float16
    #
    # map from ftype to string
    ftype_str = ["f32", "f16"]

    ftype = 1
    # if args.use_f32:
    #    ftype = 0

    model = AutoModelForCausalLM.from_pretrained(
        dir_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to('cpu')

    fname_middle = "evaclip_"
    has_text_encoder = False
    has_vision_encoder = True
    has_llava_projector = True

    output_dir = args.output_dir if args.output_dir is not None else dir_model
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.basename(output_dir).replace("ggml_", "")
    fname_out = os.path.join(output_dir, f"{fname_middle}model-{ftype_str[ftype]}.gguf")
    fout = GGUFWriter(path=fname_out, arch="evaclip")

    fout.add_bool("clip.has_text_encoder", has_text_encoder)
    fout.add_bool("clip.has_vision_encoder", has_vision_encoder)
    fout.add_bool("clip.has_llava_projector", has_llava_projector)
    fout.add_file_type(ftype)
    model_name = config["_name_or_path"] if "_name_or_path" in config else os.path.basename(dir_model)
    fout.add_name(model_name)
    fout.add_description("EVA-CLIP model")

    # vision_model hparams
    fout.add_uint32("clip.vision.image_size", v_hparams["image_size"])
    fout.add_uint32("clip.vision.patch_size", v_hparams["patch_size"])
    fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), v_hparams["hidden_size"])
    fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), v_hparams["intermediate_size"])
    fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), v_hparams["num_heads"])
    fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), v_hparams["layer_norm_eps"])
    block_count = v_hparams["num_hidden_layers"] if has_llava_projector else v_hparams["num_hidden_layers"]
    fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), block_count)
    #
    #     if processor is not None:
    #         image_mean = processor.image_processor.image_mean if args.image_mean is None or args.image_mean == default_image_mean else args.image_mean
    #         image_std = processor.image_processor.image_std if args.image_std is None or args.image_std == default_image_std else args.image_std
    #     else:
    #         image_mean = args.image_mean if args.image_mean is not None else default_image_mean
    #         image_std = args.image_std if args.image_std is not None else default_image_std
    default_image_mean = [0.48145466, 0.4578275, 0.40821073]
    default_image_std = [0.26862954, 0.26130258, 0.27577711]
    fout.add_array("clip.vision.image_mean", default_image_mean)
    fout.add_array("clip.vision.image_std", default_image_std)

    use_gelu = True
    fout.add_bool("clip.use_gelu", use_gelu)

    for l in range(block_count):
        for layer_spec in vision_tensor_name_mapping_layers:
            py_name = layer_spec[0].format(l)
            if layer_spec[2]:
                llama_names = [layer_spec[1].format(l, a) for a in layer_spec[3]]
                vision_extra_mapping_args[py_name] = llama_names
                continue

            llama_name = layer_spec[1].format(l)
            tensor_name_mapping[py_name] = llama_name

    state_dict = model.state_dict()
    for name, data in state_dict.items():
        mapped_name = tensor_name_mapping.get(name, None)
        if mapped_name is None:
            vision_extra_args = vision_extra_mapping_args.get(name, None)
            language_extra_args = language_extra_mapping_args.get(name, None)
            if vision_extra_args:
                if "weight" in name:
                    data = data.reshape(3, v_hparams["hidden_size"], v_hparams["hidden_size"]).float().squeeze().numpy()
                elif "bias" in name:
                    data = data.reshape(3, v_hparams["hidden_size"]).float().squeeze().numpy()
                for idx, e in enumerate(vision_extra_args):
                    fout.add_tensor(e, data[idx, ...])
                    print(f"tensor {e} is always saved in f16")
                continue
            else:
                continue

        name = mapped_name

        data = data.float().squeeze().numpy()

        n_dims = len(data.shape)

        # ftype == 0 -> float32, ftype == 1 -> float16
        ftype_cur = 0
        if n_dims == 4:
            print(f"tensor {name} is always saved in f16")
            data = data.astype(np.float16)
            ftype_cur = 1
        elif ftype == 1:
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

        print(f"{name} - {ftype_str[ftype_cur]} - shape = {data.shape}")
        fout.add_tensor(name, data)

    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()

    print("Done. Output file: " + fname_out)
