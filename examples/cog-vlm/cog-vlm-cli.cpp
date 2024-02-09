#include "ggml.h"
#include "common.h"
#include "eva-clip.h"
#include "cog-vlm.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    llama_set_current_expert(ctx_llama, 0);

    int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0))) {
            fprintf(stderr, "%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, bool add_bos){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
    return true;
}

static const char * sample(struct llama_sampling_context * ctx_sampling,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = llama_sampling_sample(ctx_sampling, ctx_llama, NULL);
    llama_sampling_accept(ctx_sampling, ctx_llama, id, true);
    static std::string ret;
    if (id == llama_token_eos(llama_get_model(ctx_llama))) {
        ret = "</s>";
    } else {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}

struct cog_vlm_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

static void show_additional_info(int /*argc*/, char ** argv) {
    fprintf(stderr, "\n example usage: %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    fprintf(stderr, "  note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static struct llava_image_embed * load_image(cog_vlm_context * ctx_llava, gpt_params * params) {

    // load and preprocess the image
    llava_image_embed * embed = NULL;
    embed = llava_image_embed_make_with_filename(ctx_llava->ctx_clip, params->n_threads, params->image.c_str());
    if (!embed) {
        fprintf(stderr, "%s: is %s really an image file?\n", __func__, params->image.c_str());
        return NULL;
    }

    return embed;
}

static void process_prompt(struct cog_vlm_context * ctx_llava, struct llava_image_embed * image_embed, gpt_params * params, const std::string & prompt) {
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;
    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx_llava->ctx_llama));

    eval_string(ctx_llava->ctx_llama, "", params->n_batch, &n_past, add_bos);

    if (image_embed != nullptr) {
        llava_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past);
    }

    eval_string(ctx_llava->ctx_llama, prompt .c_str(), params->n_batch, &n_past, false);

    // generate the response

    fprintf(stderr, "\n");

    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params->sparams);

    for (int i = 0; i < max_tgt_len; i++) {
        const char * tmp = sample(ctx_sampling, ctx_llava->ctx_llama, &n_past);
        if (strcmp(tmp, "</s>") == 0) break;

        printf("%s", tmp);
        fflush(stdout);
    }

    llama_sampling_free(ctx_sampling);
    printf("\n");
}


static struct cog_vlm_context * cog_vlm_init(gpt_params * params) {
    const char * clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "die";
    }

    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);

    llama_backend_init(params->numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(*params);

    llama_model * model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return NULL;
    }


    llama_context_params ctx_params = llama_context_params_from_gpt_params(*params);
    ctx_params.n_ctx           = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);
    if (ctx_llama == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return NULL;
    }

    auto ctx_cvlm = (struct cog_vlm_context *)malloc(sizeof(cog_vlm_context));

    ctx_cvlm->ctx_clip = ctx_clip;
    ctx_cvlm->ctx_llama = ctx_llama;
    ctx_cvlm->model = model;
    return ctx_cvlm;
}

static void llava_free(struct cog_vlm_context * ctx_llava) {
    if (ctx_llava->ctx_clip) {
        clip_free(ctx_llava->ctx_clip);
        ctx_llava->ctx_clip = NULL;
    }

    llama_free(ctx_llava->ctx_llama);
    llama_free_model(ctx_llava->model);
    llama_backend_free();
}

int main(int argc, char ** argv) {
    ggml_time_init();

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        show_additional_info(argc, argv);
        return 1;
    }

    auto ctx_llava = cog_vlm_init(&params);
    if (ctx_llava == NULL) {
        fprintf(stderr, "%s: error: failed to init cogvlm\n", __func__);
        return 1;
    }

    auto image_embed = load_image(ctx_llava, &params);
    if (!image_embed) {
        return 1;
    }

    // process the prompt
    //process_prompt(ctx_llava, image_embed, &params, params.prompt);

    //llama_print_timings(ctx_llava->ctx_llama);

    llava_image_embed_free(image_embed);
    llava_free(ctx_llava);

    return 0;
}
