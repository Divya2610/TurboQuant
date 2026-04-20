# TurboQuant KV Cache Benchmark Suite
### Gemma 4 E2B & E4B — CPU & GPU Benchmarking with Quantized KV Cache

---

## Overview

This suite benchmarks Google's Gemma 4 language models (E2B and E4B variants) using 
**TurboQuant** — a custom KV cache compression system that uses rotation-based quantization and Lloyd-Max codebooks to dramatically reduce memory usage during inference.

**Key metrics measured:**
- TTFT (Time To First Token)
- Total generation latency
- Throughput (tokens/second)
- Peak RAM usage
- Peak VRAM usage (GPU scripts only)
- KV cache compression ratio

---

## What is TurboQuant?

TurboQuant compresses the Key-Value (KV) cache produced during transformer inference using three steps:

1. **Rotation** — Each layer's KV tensors are multiplied by a random orthonormal matrix (QR decomposition), which spreads information evenly across dimensions and makes quantization more uniform.

2. **Scale normalization** — Each token vector is divided by its own standard deviation before quantization, and the scale is stored separately so the original magnitude can be recovered.

3. **Lloyd-Max quantization** — Optimal centroid positions are fitted on a Gaussian distribution, placing more centroids near zero (where data is dense) and fewer in the tails — minimizing average quantization error.

### Why asymmetric bits (3-bit keys, 2-bit values)?

| Component | Bits | Reason |
|-----------|------|--------|
| Keys | 3-bit (8 centroids) | Used in attention dot products — more sensitive to error |
| Values | 2-bit (4 centroids) | Retrieved after attention weighting — smoother, tolerates more compression |

This yields a **~12× compression ratio** over float32, or ~8× over float16.

---

## Repository Structure

```
.
├── benchmark.py                          # Single-model CPU benchmark (E2B)
├── benchmark_gemma4_e4b_cpu.py           # Single-model CPU benchmark (E4B)
├── benchmark_script_gemma2b_gpu.py       # Single-model GPU benchmark (E2B)
├── benchmark_gemma4_e4b_gpu.py           # Single-model GPU benchmark (E4B)
├── comparative_analysis_cpu.py           # Side-by-side CPU comparison (E2B vs E4B)
├── benchmark_compare_gpu.py              # Side-by-side GPU comparison (E2B vs E4B)
└── README.md
```

---

## Requirements

### Python
Python 3.9 or higher recommended.

### Dependencies
```bash
pip install torch transformers accelerate psutil
```

> For GPU scripts, ensure you have a CUDA-capable GPU and the appropriate CUDA version of PyTorch installed. Visit [pytorch.org](https://pytorch.org) for the correct install command for your system.

### Models
You need local copies of the Gemma 4 models. The scripts expect them at these default paths (configurable via CLI flags):

| Model | Default Path |
|-------|-------------|
| Gemma 4 E2B | `./google-gemma-4-E2B` |
| Gemma 4 E4B | `/home/kanshika/Desktop/Model/gemma-4-E4B` |

---
---

## CLI Arguments

All scripts share a common set of arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--runs` / `-r` | `3` | Number of timed benchmark runs (plus 1 warmup, always discarded) |
| `--prompt` / `-p` | ML explanation prompt | The text prompt used for generation |
| `--key-bits` | `3` | Bits per key element (1–8). Higher = better quality, less compression |
| `--val-bits` | `2` or `3` | Bits per value element (1–8). Higher = better quality, less compression |

**Comparison scripts only:**

| Argument | Description |
|----------|-------------|
| `--skip-e2b` | Skip E2B, run E4B only |
| `--skip-e4b` | Skip E4B, run E2B only |
| `--csv` | Output CSV file path (default: `benchmark_compare_results.csv`) |

---

## Benchmark Settings Explained

| Setting | Value | Why |
|---------|-------|-----|
| Context window | 2048 tokens | Fixed for fair comparison; large enough to make KV cache meaningful |
| Max new tokens | 128 | Enough for stable TPS measurement without hours of CPU time |
| Warmup run | 1 (discarded) | Eliminates JIT compilation and memory-mapping cold-start from Run 1 |
| Precision (CPU) | float32 | Native CPU precision; float16 has no hardware acceleration on CPU |
| Precision (GPU) | float16 | Standard GPU inference precision; halves VRAM vs float32 |
| Lloyd-Max samples | 100,000 | Sufficient to accurately estimate optimal centroid positions |
| Lloyd-Max iterations | 150 | Converges well before this limit for standard bit widths |

---

## Sample Output (CPU Comparison)

```
════════════════════════════════════════════════════════════════════════════════
  COMPARATIVE ANALYSIS  :  Gemma 4 E2B  vs  Gemma 4 E4B
  TurboQuant KV Cache  ·  key=3b  val=2b  ·  context=2048  ·  CPU / float32
════════════════════════════════════════════════════════════════════════════════

  LATENCY METRICS                E2B mean      E4B mean      Δ (E4B vs E2B)
  TTFT (s)                      0.8854s        1.5104s       ▲ +0.625  (70.6%)
  Total latency (s)            48.1814s       74.5290s       ▲ +26.35  (54.7%)

  THROUGHPUT
  Throughput (tok/s)              2.660          1.718        ▼ +0.942  (35.4%)

  KV CACHE COMPRESSION             E2B           E4B
  Float32 KV size (MB)           140.0         336.0         baseline
  TurboQuant KV (MB)             11.48         27.56         key=3b val=2b
  Compression ratio              12.19         12.19         fp32 ÷ turbo

  VERDICT
  ⏱  Faster TTFT      → E2B  (70.6% gap)
  🕐  Lower latency   → E2B  (54.7% gap)
  ⚡  Higher TPS      → E2B  (35.4% gap)
  💾  Lower RAM       → E2B  (60.1% gap)
```

---

## Output Files

Each script saves a CSV file after completion:

| Script | CSV Output |
|--------|-----------|
| `benchmark.py` (E2B CPU) | `benchmark_results.csv` |
| `benchmark_gemma4_e4b_cpu.py` | `benchmark_results_gemma4_e4b.csv` |
| `benchmark_script_gemma2b_gpu.py` | `benchmark_results_gpu.csv` |
| `benchmark_gemma4_e4b_gpu.py` | `benchmark_results_gemma4_e4b_gpu.csv` |
| `comparative_analysis_cpu.py` | `benchmark_compare_results.csv` |
| `benchmark_compare_gpu.py` | `benchmark_compare_results.csv` |

CSV files include per-run raw data, summary statistics, peak memory, and KV cache compression figures.

---

## Understanding the Results

### TTFT (Time To First Token)
The time to process all input tokens in one forward pass (the "prefill" phase). Scales with model size — E4B's 42 layers vs E2B's 35 means it's consistently slower here.

### Total Latency
Full time to generate all 128 output tokens. Dominated by the decode loop, which runs compress → decompress → forward pass 127 times. On CPU this is inherently slow.

### Throughput (tok/s)
Tokens generated per second. On CPU with TurboQuant, expect 1–5 tok/s. On GPU, expect 10–80× faster depending on hardware.

### Compression Ratio
Both models achieve the **same ratio** (e.g. 12.19×) because the ratio depends only on the bit-width configuration, not the model size. E4B saves more raw MB simply because it starts with a larger cache.

### RAM vs VRAM
- **RAM**: System memory — dominated by model weights (~20 GB for E2B, ~32 GB for E4B in float32)
- **VRAM**: GPU memory — only relevant for GPU scripts; float16 halves weight memory vs float32

---

## Known Limitations

- **Codebook fitted on synthetic data**: Lloyd-Max centroids are optimized for a Gaussian distribution, not actual KV tensors. Post-rotation KV values are approximately Gaussian, so this works well in practice, but a calibration pass on real data would give marginally tighter compression.
- **CPU is very slow**: These models were designed for GPU inference. CPU runs are useful for correctness testing and hardware-constrained environments, but production use requires a GPU.
- **uint8 index storage**: Quantization indices are stored as `uint8`, limiting bit widths to a maximum of 8 bits (256 levels). Attempting more than 8 bits raises a `ValueError`.
- **No quality evaluation**: This benchmark measures speed and memory only. It does not evaluate whether the compressed KV cache degrades output quality. Lower bit widths (especially 1–2 bits) may produce noticeably worse text.

---

## Tips

**To reduce memory usage:**
- Lower `--val-bits` to `1` (most aggressive, may hurt quality)
- Lower `--key-bits` to `2` (moderate reduction, more quality impact than values)

**To improve output quality:**
- Raise `--key-bits` to `4` and `--val-bits` to `3`
- Note: higher bits = less compression and more VRAM/RAM used for the cache

**To run on GPU with less VRAM:**
- Use float16 (default for GPU scripts)
- Reduce `--key-bits` and `--val-bits` to free more VRAM for longer contexts

**To get stable benchmark numbers:**
- Use `--runs 5` for more statistical stability
- Close other applications to reduce OS interference (especially on CPU)
- The warmup run is always added automatically — you don't need to add it yourself

---



---

