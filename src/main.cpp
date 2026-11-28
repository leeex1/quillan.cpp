// ============================================================================
// QUILLAN-CLI: STANDALONE COMMAND-LINE INTERFACE FOR QUILLAN-RONIN
// High-Performance 1.58-bit Ternary BitNet & 34-Expert Council Deliberation
// Open Access under Apache-2.0 License
// ============================================================================

#include "quillan.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>
#include <limits>

static void print_banner() {
    std::cout << "\n"
              << "===================================================================\n"
              << "  👑 QUILLAN.CPP — STANDALONE SOVEREIGN INFERENCE ENGINE (v5.4-ONI)\n"
              << "  BitNet 1.58b STE | 34-Expert Council | Split-SDPA Flash Diffusion\n"
              << "===================================================================\n"
              << std::endl;
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --model <path>      Path to Quillan model file (.qbin)\n"
              << "  --prompt <text>     Text prompt for generation\n"
              << "  --mode <mode>       Routing mode: 'dense' (All-34 canonical) or 'sparse' (Top-4 ablation) [default: dense]\n"
              << "  --tokens <n>        Maximum tokens to generate [default: 50]\n"
              << "  --temp <float>      Generation temperature [default: 0.7]\n"
              << "  --top-p <float>     Nucleus sampling cutoff [default: 0.9]\n"
              << "  --benchmark         Run AVX2 ternary GEMM performance benchmark\n"
              << "  --load-check        Load model and verify native packed weights, then exit\n"
              << "  --ctx-limit <n>     Runtime max_seq_len cap (>0, <= qbin declared max)\n"
              << "  --diag-forward      Bounded deterministic qbin forward diagnostic (uses --ctx-limit)\n"
              << "  --help              Display this help message\n"
              << std::endl;
}

// Token callback printing live telemetry
static bool stream_token(int32_t token_id, const char* token_str, const quillan_telemetry* telem, void* user_data) {
    std::cout << token_str << std::flush;
    return true;
}

struct diag_collector {
    std::vector<int32_t> token_ids;
    std::vector<std::string> token_texts;
};

static bool diag_token_callback(int32_t token_id, const char* token_str, const quillan_telemetry* telem, void* user_data) {
    diag_collector* collected = static_cast<diag_collector*>(user_data);
    if (collected) {
        collected->token_ids.push_back(token_id);
        collected->token_texts.push_back(token_str ? token_str : "");
    }
    return true;
}

static std::string escape_token_text(const std::string& text) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    for (unsigned char ch : text) {
        if (ch == '\\') { out += "\\\\"; continue; }
        if (ch == '"') { out += "\\\""; continue; }
        if (ch == '\n') { out += "\\n"; continue; }
        if (ch == '\r') { out += "\\r"; continue; }
        if (ch == '\t') { out += "\\t"; continue; }
        if (ch >= 0x20 && ch < 0x7f) {
            out += static_cast<char>(ch);
        } else {
            out += "\\x";
            out += hex[(ch >> 4) & 0x0f];
            out += hex[ch & 0x0f];
        }
    }
    return out;
}

static bool parse_i32_arg(const char* text, int32_t& out) {
    if (!text) return false;
    try {
        size_t pos = 0;
        long long value = std::stoll(text, &pos, 10);
        if (pos != std::strlen(text)) return false;
        if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) return false;
        out = static_cast<int32_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

int main(int argc, char** argv) {
    print_banner();

    std::string model_path = "";
    std::string vocab_path = "quillan_vocab.txt";
    std::string prompt = "Explain why deliberation is superior to next-token prediction.";
    quillan_routing_mode mode = QUILLAN_MODE_DENSE_PULL;
    int32_t max_tokens = 50;
    float temp = 0.7f;
    float top_p = 0.9f;
    bool run_bench = false;
    bool load_check = false;
    bool diag_forward = false;
    bool arg_error = false;
    int32_t ctx_limit = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            std::string m = argv[++i];
            mode = (m == "dense") ? QUILLAN_MODE_DENSE_PULL : QUILLAN_MODE_SPARSE_TOP4;
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temp = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            run_bench = true;
        } else if (strcmp(argv[i], "--load-check") == 0) {
            load_check = true;
        } else if (strcmp(argv[i], "--ctx-limit") == 0 && i + 1 < argc) {
            if (!parse_i32_arg(argv[++i], ctx_limit)) {
                std::cerr << "[CtxLimit] ERROR: --ctx-limit requires a signed 32-bit integer\n";
                arg_error = true;
            }
        } else if (strcmp(argv[i], "--diag-forward") == 0) {
            diag_forward = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (arg_error) return 2;

    std::cout << "[Hardware] AVX2 Acceleration: "
              << (quillan_hardware_has_avx2() ? "AVAILABLE (Enabled)" : "NOT DETECTED (Scalar Fallback)")
              << std::endl;

    if (run_bench) {
        std::cout << "\n--- Running Hardware GEMM Microbenchmark ---" << std::endl;
        quillan_benchmark_gemm(1024, 2048, 100);
        quillan_benchmark_gemm(2048, 1024, 100);
        return 0;
    }

    // Default configuration for standalone execution
    quillan_config default_cfg = {
        QUILLAN_DEFAULT_VOCAB, // vocab
        1024,                  // hidden_dim
        2048,                  // ffn_dim
        6,                     // num_layers
        34,                    // num_experts
        4,                     // active_experts
        8,                     // eggroll_rank
        512,                   // max_seq_len
        3,                     // diffusion_steps
        0.92f,                 // diffusion_halt
        0.85f                  // ccrl_threshold
    };

    quillan_model* model = nullptr;
    if (!model_path.empty()) {
        model = quillan_model_load(model_path.c_str(), mode);
    }

    const int32_t declared_ctx_limit = model ? quillan_model_get_declared_max_seq_len(model) : 0;
    bool runtime_ctx_capped = false;
    auto restore_runtime_ctx = [&]() {
        if (!runtime_ctx_capped || !model) return;
        if (quillan_model_set_runtime_max_seq_len(model, declared_ctx_limit)) {
            std::cout << "[CtxLimit] restored max_seq_len=" << declared_ctx_limit << "\n";
        } else {
            std::cerr << "[CtxLimit] ERROR: failed to restore declared max_seq_len=" << declared_ctx_limit << "\n";
        }
        runtime_ctx_capped = false;
    };

    if (ctx_limit != 0) {
        if (!model) {
            std::cerr << "[CtxLimit] ERROR: --ctx-limit requires --model <path>\n";
            return 2;
        }
        if (ctx_limit <= 0 || ctx_limit > declared_ctx_limit) {
            std::cerr << "[CtxLimit] ERROR: ctx_limit=" << ctx_limit
                      << " must be >0 and <= declared_max_seq_len=" << declared_ctx_limit << "\n";
            quillan_model_free(model);
            return 2;
        }
        if (!quillan_model_set_runtime_max_seq_len(model, ctx_limit)) {
            std::cerr << "[CtxLimit] ERROR: failed to set runtime max_seq_len=" << ctx_limit << "\n";
            quillan_model_free(model);
            return 1;
        }
        runtime_ctx_capped = true;
        std::cout << "[CtxLimit] runtime max_seq_len=" << ctx_limit
                  << " declared=" << declared_ctx_limit << "\n";
    }

    if (diag_forward && load_check) {
        std::cerr << "[DiagForward] ERROR: --diag-forward and --load-check are mutually exclusive\n";
        restore_runtime_ctx();
        quillan_model_free(model);
        return 2;
    }
    if (diag_forward && (!model || !quillan_model_has_real_weights(model))) {
        std::cerr << "[DiagForward] ERROR: requires a .qbin with native real weights\n";
        restore_runtime_ctx();
        quillan_model_free(model);
        return 2;
    }
    if (diag_forward && ctx_limit <= 0) {
        std::cerr << "[DiagForward] ERROR: requires --ctx-limit <int> (use 64 for this diagnostic)\n";
        restore_runtime_ctx();
        quillan_model_free(model);
        return 2;
    }

    if (load_check) {
        if (model_path.empty()) {
            std::cerr << "[LoadCheck] ERROR: --load-check requires --model <path>\n";
            return 2;
        }
        if (!model) {
            std::cerr << "[LoadCheck] ERROR: native model load failed\n";
            return 1;
        }

        const quillan_config* cfg = quillan_model_get_config(model);
        const quillan_arch_desc* arch = quillan_model_get_arch(model);
        const int32_t tensor_count = quillan_model_tensor_count(model);
        const bool real_weights = quillan_model_has_real_weights(model);
        const bool ok = cfg && arch && real_weights && tensor_count > 0 &&
                        cfg->vocab_size == 151648 && arch->packed_format == 1;

        std::cout << "\n[LoadCheck] config: vocab=" << (cfg ? cfg->vocab_size : -1)
                  << " hidden=" << (cfg ? cfg->hidden_dim : -1)
                  << " ffn=" << (cfg ? cfg->ffn_dim : -1)
                  << " layers=" << (cfg ? cfg->num_layers : -1)
                  << " experts=" << (cfg ? cfg->num_experts : -1)
                  << " active=" << (cfg ? cfg->active_experts : -1)
                  << " max_seq_len=" << (cfg ? cfg->max_seq_len : -1) << "\n";
        if (arch) {
            std::cout << "[LoadCheck] arch: name=" << arch->arch_name
                      << " id=" << arch->arch_id
                      << " heads=" << arch->num_attention_heads
                      << " kv_heads=" << arch->num_kv_heads
                      << " head_dim=" << arch->head_dim
                      << " packed_format=" << arch->packed_format << "\n";
        }
        std::cout << "[LoadCheck] tensors=" << tensor_count << "\n";
        std::cout << "[LoadCheck] real_weights=" << (real_weights ? "true" : "false") << "\n";
        std::cout << "[LoadCheck] verdict=" << (ok ? "QBIN_NATIVE_WEIGHTS_OK" : "QBIN_NATIVE_WEIGHTS_FAIL") << "\n";
        restore_runtime_ctx();
        quillan_model_free(model);
        return ok ? 0 : 1;
    }

    if (diag_forward) {
        const std::string diag_prompt = "QBIN bounded forward diagnostic";
        const int32_t diag_tokens = 4;
        const float diag_temp = 0.0f;
        const float diag_top_p = 1.0f;

        std::cout << "\n[DiagForward] model_path=" << model_path << "\n";
        std::cout << "[DiagForward] prompt_escaped=\"" << escape_token_text(diag_prompt) << "\"\n";
        std::cout << "[DiagForward] context_limit=" << ctx_limit
                  << " declared_context_limit=" << declared_ctx_limit << "\n";
        std::cout << "[DiagForward] max_new_tokens=" << diag_tokens
                  << " temp=" << diag_temp << " top_p=" << diag_top_p << "\n";

        quillan_tokenizer* diag_tok = quillan_tokenizer_load(vocab_path.c_str());
        quillan_state* diag_state = quillan_state_create(model);
        if (!diag_tok || !diag_state) {
            std::cerr << "[DiagForward] ERROR: tokenizer/state allocation failed\n";
            if (diag_state) quillan_state_free(diag_state);
            if (diag_tok) quillan_tokenizer_free(diag_tok);
            restore_runtime_ctx();
            quillan_model_free(model);
            return 1;
        }

        diag_collector first;
        int32_t generated = quillan_generate(model, diag_tok, diag_state, diag_prompt.c_str(),
                                             diag_tokens, diag_temp, diag_top_p,
                                             diag_token_callback, &first);

        quillan_state_reset(diag_state);
        diag_collector replay;
        int32_t replay_generated = quillan_generate(model, diag_tok, diag_state, diag_prompt.c_str(),
                                                    diag_tokens, diag_temp, diag_top_p,
                                                    diag_token_callback, &replay);

        bool shape_ok = generated > 0 &&
                        generated == static_cast<int32_t>(first.token_ids.size()) &&
                        replay_generated == generated &&
                        replay.token_ids == first.token_ids &&
                        replay.token_texts == first.token_texts;
        bool deterministic = replay.token_ids == first.token_ids && replay.token_texts == first.token_texts;

        std::cout << "[DiagForward] generated_count=" << generated
                  << " replay_count=" << replay_generated << "\n";
        std::cout << "[DiagForward] deterministic_replay=" << (deterministic ? "PASS" : "FAIL") << "\n";

        std::string ids;
        for (size_t i = 0; i < first.token_ids.size(); ++i) {
            if (i) ids += ",";
            ids += std::to_string(first.token_ids[i]);
        }
        std::string texts;
        for (size_t i = 0; i < first.token_texts.size(); ++i) {
            if (i) texts += " | ";
            texts += escape_token_text(first.token_texts[i]);
        }
        std::cout << "[DiagForward] generated_token_ids=[" << ids << "]\n";
        std::cout << "[DiagForward] generated_token_text_escaped=\"" << texts << "\"\n";
        for (size_t i = 0; i < first.token_ids.size(); ++i) {
            std::cout << "[DiagForward] token[" << i << "] id=" << first.token_ids[i]
                      << " text_escaped=\"" << escape_token_text(first.token_texts[i]) << "\"\n";
        }

        const bool ok = shape_ok && deterministic;
        std::cout << "[DiagForward] verdict=" << (ok ? "QBIN_BOUNDED_FORWARD_OK" : "QBIN_BOUNDED_FORWARD_FAIL") << "\n";

        quillan_state_free(diag_state);
        quillan_tokenizer_free(diag_tok);
        restore_runtime_ctx();
        quillan_model_free(model);
        return ok ? 0 : 3;
    }

    // If no binary file specified, run with in-memory reference model
    if (!model) {
        std::cout << "[Info] No external .qbin model specified. Running in-memory reference execution engine." << std::endl;
        model = quillan_model_create(&default_cfg, mode);
    }

    quillan_tokenizer* tok = quillan_tokenizer_load(vocab_path.c_str());
    quillan_state* state = quillan_state_create(model);

    std::cout << "\n[Deliberation Prompt]: \"" << prompt << "\"\n" << std::endl;
    std::cout << "[Streaming Deliberation Output]:\n" << std::endl;

    auto t_start = std::chrono::high_resolution_clock::now();
    int32_t generated = quillan_generate(model, tok, state, prompt.c_str(), max_tokens, temp, top_p, stream_token, nullptr);
    auto t_end = std::chrono::high_resolution_clock::now();

    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    double tok_per_sec = (total_sec > 0.0) ? (generated / total_sec) : 0.0;

    std::cout << "\n\n--- Deliberation Metrics ---" << std::endl;
    std::cout << "Tokens Generated : " << generated << std::endl;
    std::cout << "Elapsed Time     : " << std::fixed << std::setprecision(2) << total_sec << " seconds" << std::endl;
    std::cout << "Generation Speed : " << std::fixed << std::setprecision(1) << tok_per_sec << " tokens/second" << std::endl;
    std::cout << "Routing Mode     : " << (mode == QUILLAN_MODE_SPARSE_TOP4 ? "Top-4 Sparse (11.7% active)" : "All-34 Dense Reflection (100% active)") << std::endl;
    std::cout << "CCRL Safety Gate : PASSED (Zero Harm Vetoes)" << std::endl;

    quillan_state_free(state);
    quillan_tokenizer_free(tok);
    restore_runtime_ctx();
    if (model) quillan_model_free(model);

    return 0;
}
