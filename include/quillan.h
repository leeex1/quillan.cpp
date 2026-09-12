// ============================================================================
// QUILLAN.CPP: STANDALONE C++17 INFERENCE ENGINE FOR QUILLAN-RONIN
// High-Performance 1.58-bit Ternary BitNet, 9-Vector Prism & Council MoE
// Header-Only Core API Definition
// Open Access under Apache-2.0 License
// ============================================================================

#ifndef QUILLAN_H
#define QUILLAN_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QUILLAN_MAGIC 0x4E4C4C51 // "QLLN" in little-endian
#define QUILLAN_VERSION 1
#define QUILLAN_DEFAULT_VOCAB 50257
#define QUILLAN_NUM_EXPERTS 34
#define QUILLAN_NUM_RAYS 9

// Execution routing modes
typedef enum {
    QUILLAN_MODE_SPARSE_TOP4 = 0, // 4 of 34 active (~11.7% parameters)
    QUILLAN_MODE_DENSE_PULL   = 1  // All 34 active (full reflection consensus)
} quillan_routing_mode;

// Configuration structure
typedef struct {
    int32_t vocab_size;
    int32_t hidden_dim;       // e.g. 1024 (Oni) or 2560 (Saturated)
    int32_t ffn_dim;          // e.g. 2048
    int32_t num_layers;       // e.g. 6 (Proof) or 12 (Flagship)
    int32_t num_experts;      // 34
    int32_t active_experts;   // 4 (sparse) or 34 (dense)
    int32_t eggroll_rank;     // 8 (Oni) or 24 (Saturated)
    int32_t max_seq_len;      // 512 or 1024
    int32_t diffusion_steps;  // Max Langevin steps (e.g. 3)
    float   diffusion_halt;   // Confidence exit threshold (0.92)
    float   ccrl_threshold;   // Bayesian safety veto threshold (0.85)
} quillan_config;

// Deliberation runtime diagnostics
typedef struct {
    float prism_rays[QUILLAN_NUM_RAYS];   // Activation norm per semantic ray
    int32_t active_experts[QUILLAN_NUM_EXPERTS]; // Indices of routed experts
    float expert_weights[QUILLAN_NUM_EXPERTS];  // Routing gate weights
    int32_t num_active_experts;
    int32_t diffusion_steps_taken;
    float confidence_score;
    float ccrl_score;
    bool ccrl_vetoed;
    double token_latency_ms;
} quillan_telemetry;

// Opaque handles
typedef struct quillan_model quillan_model;
typedef struct quillan_tokenizer quillan_tokenizer;
typedef struct quillan_state quillan_state;

// Callback for streaming generation
typedef bool (*quillan_token_callback)(int32_t token_id, const char* token_str, const quillan_telemetry* telem, void* user_data);

// ----------------------------------------------------------------------------
// Public API Functions
// ----------------------------------------------------------------------------

// Model lifecycle
quillan_model* quillan_model_create(const quillan_config* config, quillan_routing_mode mode);
quillan_model* quillan_model_load(const char* model_path, quillan_routing_mode mode);
void quillan_model_free(quillan_model* model);
const quillan_config* quillan_model_get_config(const quillan_model* model);

// Tokenizer lifecycle
quillan_tokenizer* quillan_tokenizer_load(const char* vocab_path);
void quillan_tokenizer_free(quillan_tokenizer* tok);
int32_t quillan_tokenize(const quillan_tokenizer* tok, const char* text, int32_t* tokens, int32_t max_tokens);
int32_t quillan_detokenize(const quillan_tokenizer* tok, int32_t token, char* buf, int32_t max_len);

// Execution State & Deliberation
quillan_state* quillan_state_create(const quillan_model* model);
void quillan_state_reset(quillan_state* state);
void quillan_state_free(quillan_state* state);

// Single token deliberation step
int32_t quillan_deliberate_step(
    quillan_model* model,
    quillan_state* state,
    int32_t input_token,
    float temperature,
    float top_p,
    quillan_telemetry* out_telem
);

// High-level text generation
int32_t quillan_generate(
    quillan_model* model,
    const quillan_tokenizer* tok,
    quillan_state* state,
    const char* prompt,
    int32_t max_new_tokens,
    float temperature,
    float top_p,
    quillan_token_callback callback,
    void* user_data
);

// Hardware & Benchmark Telemetry
bool quillan_hardware_has_avx2(void);
void quillan_benchmark_gemm(int32_t rows, int32_t cols, int32_t iterations);

#ifdef __cplusplus
}
#endif

#endif // QUILLAN_H
