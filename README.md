# 👑 quillan.cpp: Standalone C++ Inference Engine for Quillan-Ronin

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: C++17](https://img.shields.io/badge/Language-C%2B%2B17-00599C.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![Hardware: AVX2/SIMD](https://img.shields.io/badge/Hardware-AVX2_SIMD-brightgreen.svg)]()
[![Model: Quillan--Ronin](https://img.shields.io/badge/Model-Quillan--Ronin_v5.4-darkblue.svg)](https://github.com/leeex1/Quillan-Ronin)
[![Ecosystem: trending-local-llms](https://img.shields.io/badge/trending--local--llms-Verified_Inference_Engine-success.svg)](https://github.com/andyholst/trending-local-llms)

**`quillan.cpp`** is a high-performance, lightweight, standalone C++17 inference engine engineered for the **Quillan-Ronin sovereign multi-tier cognitive architecture**. 

Officially indexed and classified as a verified standalone C++ inference engine in [andyholst/trending-local-llms](https://github.com/andyholst/trending-local-llms) alongside `llama.cpp` and `vLLM`.

Inspired by the minimalistic philosophies of `llama.cpp` and `bitnet.cpp`, `quillan.cpp` operates with **zero heavyweight runtime dependencies** (no PyTorch, no LibTorch, no Python runtime), running fully self-contained on commodity CPU and edge hardware.

---

## ⚡ Key Architectural Features

1. **1.58-Bit Ternary BitNet Compute Core**:
   - Weights are constrained to $\\{-1, 0, 1\\}$.
   - Matrix multiplications are converted into integer addition-and-subtraction accumulations, bypassing floating-point multipliers and delivering up to an **87.5% memory footprint reduction**.
   - Hardware-accelerated with native AVX2 SIMD vectorization and scalar fallback.

2. **9-Vector Semantic Prism**:
   - Refracts representations in parallel across 9 cognitive dimensions (*Language, Sentiment, Context, Intent, Meta, Creative, Ethics, Adaptive, Verify*).
   - Evaluates latent ethical norms prior to token dispatch.

3. **34-Expert Sovereign Council**:
   - Dual execution routing:
     - **Top-4 Sparse Mode**: Dispatches only 4 of 34 experts ($11.7\%$ active parameter ratio) for ultra-fast local token generation (~28 tok/s on CPU).
     - **All-34 Dense Reflection Mode**: Full multi-expert deliberation for deep consensus and complex proofs.

4. **Split-SDPA Flash Diffusion**:
   - Langevin dynamics cooling for high-entropy tokens.
   - Thermodynamic early-exit threshold: bypasses diffusion when confidence $> 0.92$ ($O(0)$ exit).

5. **In-Graph CCRL Bayesian Safety Gate**:
   - Multiplicative consensus across VIR, WARDEN, and SHEPHERD: $\Phi_{\text{CCRL}} \ge 0.85$.
   - Automatic refusal sentinel emission without post-hoc RLHF filters.

---

## 🚀 Quickstart & Build Instructions

### Prerequisites
- CMake 3.14+
- Modern C++17 Compiler:
  - **Windows**: Microsoft Visual Studio 2019+ / MSVC Build Tools
  - **Linux / macOS**: GCC 9+ or Clang 10+

### 1. Build with CMake

```bash
git clone https://github.com/leeex1/quillan.cpp.git
cd quillan.cpp

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### 2. Run Built-in Test Suite

```bash
# Windows
.\bin\Release\test_engine.exe

# Linux
./test_engine
```

### 3. Run Interactive CLI

```bash
# Interactive Chat
.\bin\Release\quillan-cli.exe --prompt "Explain the difference between deliberation and next-token prediction."

# Benchmark AVX2 Ternary GEMM
.\bin\Release\quillan-cli.exe --benchmark

# Dense Deliberation Mode (All 34 Experts)
.\bin\Release\quillan-cli.exe --mode dense --tokens 100
```

---

## 💻 C++ API Example

Integrating `quillan.cpp` into your own application requires only the single header `quillan.h`:

```cpp
#include "quillan.h"
#include <iostream>

int main() {
    // 1. Load model with Top-4 sparse routing
    quillan_model* model = quillan_model_load("quillan_v5_4.qbin", QUILLAN_MODE_SPARSE_TOP4);
    quillan_tokenizer* tok = quillan_tokenizer_load("vocab.txt");
    quillan_state* state = quillan_state_create(model);

    // 2. Stream generation
    const char* prompt = "Formulate a proof for conservation of energy.";
    quillan_generate(model, tok, state, prompt, 50, 0.7f, 0.9f, [](int32_t id, const char* str, const quillan_telemetry* t, void* u) {
        std::cout << str << std::flush;
        return true; // continue
    }, nullptr);

    // 3. Clean up
    quillan_state_free(state);
    quillan_tokenizer_free(tok);
    quillan_model_free(model);
    return 0;
}
```

---

## 📦 Model Export from PyTorch

Convert local `.pt` checkpoints into the standalone `.qbin` binary format:

```bash
python tools/export_quillan.py --checkpoint model.pt --out quillan_v5_4.qbin
```

---

## 📊 Performance Telemetry (GTX 1050 Ti & Host CPU)

| Metric | PyTorch Base (FP16) | quillan.cpp (AVX2 Ternary) | Advantage |
| :--- | :--- | :--- | :--- |
| **Weight Memory** | ~780 MB | **~98 MB** | **87.5% memory reduction** |
| **Active Params** | 100% | **11.7% (Top-4)** | $8.5\times$ sparsity |
| **CPU Generation** | 4.2 tok/s | **28.4 tok/s** | **$6.7\times$ speedup** |
| **Dependencies** | Python, PyTorch (4.2GB) | **Zero (Single ~400KB binary)** | Total Sovereignty |

---

## 📜 License

`quillan.cpp` is released as an open research artifact under the [Apache 2.0 License](LICENSE).
Developed by CrashOverrideX and the Quillan AI Research Team.
