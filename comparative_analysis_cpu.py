#!/usr/bin/env python3
"""
benchmark_compare.py — Gemma 4 E2B vs E4B · TurboQuant KV Cache · CPU-only
============================================================================
Runs both models back-to-back and prints a detailed side-by-side comparison.

Metrics  :  TTFT  |  Latency  |  Throughput  |  RAM (GB)  |  KV Compression
Context  :  2 048 tokens (fixed)
Backend  :  CPU only, float32
Models   :  google/gemma-4-E2B  (35 layers)
            google/gemma-4-E4B  (42 layers)

Usage:
    python benchmark_compare.py \
        --e2b-model ./google-gemma-4-E2B \
        --e4b-model /home/kanshika/Desktop/Model/gemma-4-E4B \
        --runs 3 \
        --key-bits 3 \
        --val-bits 2

Fixes applied vs original:
  1.  torch_dtype= kwarg (not dtype=) used in from_pretrained — the wrong
      keyword was silently ignored, loading the model in default precision.
  2.  TurboQuantEngine now stores self.num_layers, self.num_kv_heads —
      previously num_layers was a local variable only; any external reference
      (and the compression estimate) would raise AttributeError.
  3.  Compression size formula corrected to include num_kv_heads × head_dim
      and scale-factor overhead — the old formula was a gross undercount that
      made compression ratios look far too high.
  4.  save_csv run numbering switched from list.index() to enumerate() —
      list.index() returns the first matching index, so duplicate metric
      values would produce wrong run numbers.
  5.  Warmup run added before timed runs — eliminates JIT / memory-mapping
      overhead from Run 1, consistent with the GPU benchmark script.
  6.  Decode-mask length verified each step — guards against silent
      misalignment across transformers versions (mirrors GPU script Fix 8).
  7.  MODEL_SPECS layer counts removed — layers are now read from model
      config at runtime so they are always accurate for any variant.
"""

import argparse
import csv
import gc
import os
import statistics
import sys
import threading
import time

# ── dependency checks ──────────────────────────────────────────────────────────
try:
    import psutil
except ImportError:
    sys.exit("❌  pip install psutil")

try:
    import torch
except ImportError:
    sys.exit("❌  pip install torch")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
except ImportError:
    sys.exit("❌  pip install transformers accelerate")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CONTEXT_WINDOW   = 2048
MAX_NEW_TOKENS   = 128
DEFAULT_RUNS     = 3
DEFAULT_KEY_BITS = 3
DEFAULT_VAL_BITS = 2

# FIX 7: layer counts removed — read from model config at runtime
MODEL_SPECS = {
    "E2B": {"label": "Gemma 4 E2B", "default_path": "./google-gemma-4-E2B"},
    "E4B": {"label": "Gemma 4 E4B", "default_path": "/home/kanshika/Desktop/Model/gemma-4-E4B"},
}

DEFAULT_PROMPT = (
    "Explain the key differences between supervised, unsupervised, and "
    "reinforcement learning. Give one real-world example for each."
)

GEMMA4_CHAT_TEMPLATE = (
    "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
)

W    = 80
SEP  = "─" * W
SEP2 = "═" * W
SEP3 = "┄" * W


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMICCACHE HELPERS  (version-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_has_old_api(cache: DynamicCache) -> bool:
    return hasattr(cache, "key_cache")


def extract_kv_from_cache(cache: DynamicCache) -> list:
    if _cache_has_old_api(cache):
        return [(k, v) if k is not None else None
                for k, v in zip(cache.key_cache, cache.value_cache)]
    result = []
    for layer in cache.layers:
        if layer is None:
            result.append(None)
        else:
            k = getattr(layer, "key", None)
            v = getattr(layer, "value", None)
            if k is None:
                try:
                    k, v = layer.key_cache, layer.value_cache
                except AttributeError:
                    result.append(None)
                    continue
            result.append((k, v))
    return result


def build_cache_from_kv(layer_kvs: list) -> DynamicCache:
    new_cache = DynamicCache()
    for i, kv in enumerate(layer_kvs):
        if kv is not None:
            new_cache.update(kv[0], kv[1], i)
    return new_cache


# ─────────────────────────────────────────────────────────────────────────────
# RAM SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

class RAMSampler:
    def __init__(self):
        self._proc   = psutil.Process(os.getpid())
        self._peak   = 0.0
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss / (1024 ** 3)
            if rss > self._peak:
                self._peak = rss
            time.sleep(0.1)

    def start(self):
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return self._peak

    @property
    def current_gb(self) -> float:
        return self._proc.memory_info().rss / (1024 ** 3)


# ─────────────────────────────────────────────────────────────────────────────
# TURBOQUANT CORE
# ─────────────────────────────────────────────────────────────────────────────

def build_rotation_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    G = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q


def fit_lloyd_max(n_bits: int, n_samples: int = 100_000, n_iter: int = 150) -> tuple:
    n_levels  = 2 ** n_bits
    gen       = torch.Generator()
    gen.manual_seed(0)
    sample    = torch.randn(n_samples, generator=gen)
    centroids = torch.linspace(-3.0, 3.0, n_levels)

    for _ in range(n_iter):
        boundaries  = (centroids[:-1] + centroids[1:]) / 2.0
        full_bounds = torch.cat([torch.tensor([-1e9]), boundaries, torch.tensor([1e9])])
        new_c       = torch.empty_like(centroids)
        for k in range(n_levels):
            mask     = (sample >= full_bounds[k]) & (sample < full_bounds[k + 1])
            new_c[k] = sample[mask].mean() if mask.any() else centroids[k]
        if (new_c - centroids).abs().max().item() < 1e-7:
            break
        centroids = new_c

    return boundaries, centroids


# ─────────────────────────────────────────────────────────────────────────────
# PER-LAYER COMPRESSOR
# ─────────────────────────────────────────────────────────────────────────────

class LayerCompressor:
    def __init__(self, head_dim: int, key_bits: int, val_bits: int, layer_idx: int):
        self.head_dim = head_dim
        self.Q_k      = build_rotation_matrix(head_dim, seed=layer_idx * 2)
        self.Q_v      = build_rotation_matrix(head_dim, seed=layer_idx * 2 + 1)
        self.k_bounds = self.k_cents = self.v_bounds = self.v_cents = None

    def _compress(self, x, Q, bounds):
        y      = x @ Q.T
        scale  = y.std(dim=-1, keepdim=True).clamp(min=1e-8)
        idx    = torch.bucketize((y / scale).contiguous(), bounds)
        return idx.to(torch.uint8), scale

    def _decompress(self, idx, scale, Q, cents):
        return (cents[idx.long()] * scale) @ Q

    def compress(self, key, value):
        k_idx, k_scale = self._compress(key,   self.Q_k, self.k_bounds)
        v_idx, v_scale = self._compress(value, self.Q_v, self.v_bounds)
        return {"k_idx": k_idx, "k_scale": k_scale,
                "v_idx": v_idx, "v_scale": v_scale}

    def decompress(self, data):
        key   = self._decompress(data["k_idx"], data["k_scale"], self.Q_k, self.k_cents)
        value = self._decompress(data["v_idx"], data["v_scale"], self.Q_v, self.v_cents)
        return key, value


# ─────────────────────────────────────────────────────────────────────────────
# TURBOQUANT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TurboQuantEngine:
    def __init__(self, model, key_bits: int, val_bits: int):
        cfg      = model.config
        text_cfg = getattr(cfg, "text_config", cfg)

        # FIX 2: store as instance attributes — previously num_layers was a
        # local variable only, causing AttributeError on any external access
        self.num_layers = text_cfg.num_hidden_layers
        self.key_bits   = key_bits
        self.val_bits   = val_bits

        head_dim = getattr(text_cfg, "head_dim", None)
        if head_dim is None:
            head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
        self.head_dim = head_dim

        # FIX 2 + 3: store num_kv_heads for accurate compression estimates
        self.num_kv_heads = getattr(
            text_cfg, "num_key_value_heads",
            text_cfg.num_attention_heads,
        )

        print(f"  [TurboQuant] Fitting Lloyd-Max codebooks …  "
              f"(key={key_bits}b/{2**key_bits}c  val={val_bits}b/{2**val_bits}c)")
        t0 = time.perf_counter()
        k_bounds, k_cents = fit_lloyd_max(key_bits)
        v_bounds, v_cents = fit_lloyd_max(val_bits)
        print(f"  [TurboQuant] Codebooks done in {time.perf_counter() - t0:.2f}s")

        self.compressors = []
        for i in range(self.num_layers):
            lc = LayerCompressor(head_dim, key_bits, val_bits, layer_idx=i)
            lc.k_bounds, lc.k_cents = k_bounds, k_cents
            lc.v_bounds, lc.v_cents = v_bounds, v_cents
            self.compressors.append(lc)
        print(f"  [TurboQuant] {self.num_layers} layer compressors ready "
              f"(head_dim={head_dim}, kv_heads={self.num_kv_heads})\n")

    def compress_cache(self, cache: DynamicCache) -> list:
        layer_kvs  = extract_kv_from_cache(cache)
        compressed = []
        for i, cmp in enumerate(self.compressors):
            if i < len(layer_kvs) and layer_kvs[i] is not None:
                compressed.append(cmp.compress(*layer_kvs[i]))
            else:
                compressed.append(None)
        return compressed

    def decompress_to_cache(self, compressed: list) -> DynamicCache:
        layer_kvs = []
        for cmp, data in zip(self.compressors, compressed):
            layer_kvs.append(cmp.decompress(data) if data is not None else None)
        return build_cache_from_kv(layer_kvs)

    # FIX 3: accurate size estimate — includes num_kv_heads × head_dim and
    # scale-factor overhead, matching the formula in the GPU benchmark script
    def estimate_sizes_mb(self, context_len: int) -> dict:
        h   = self.num_kv_heads
        d   = self.head_dim
        L   = self.num_layers
        seq = context_len

        # float32 baseline: K + V, 4 bytes each
        float32_bytes = seq * h * d * 2 * 4 * L
        k_idx_bytes   = seq * h * d * (self.key_bits / 8) * L
        v_idx_bytes   = seq * h * d * (self.val_bits / 8) * L
        # scale factors: one float32 scalar per (token, kv_head) for K and V
        scale_bytes   = seq * h * 1 * 4 * 2 * L
        turbo_bytes   = k_idx_bytes + v_idx_bytes + scale_bytes

        return {
            "float32_mb": float32_bytes / (1024 ** 2),
            "turbo_mb"  : turbo_bytes   / (1024 ** 2),
            "ratio"     : float32_bytes / turbo_bytes if turbo_bytes > 0 else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE BENCHMARK RUN
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_single(model, tokenizer, input_ids, attention_mask, tq_engine) -> dict:
    eos_id           = tokenizer.eos_token_id
    tokens_generated = 0
    seq_len          = input_ids.shape[-1]
    t_start          = time.perf_counter()

    out              = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    ttft             = time.perf_counter() - t_start
    next_token       = out.logits[:, -1:, :].argmax(dim=-1)
    tokens_generated += 1

    compressed_store = tq_engine.compress_cache(out.past_key_values)
    del out
    gc.collect()

    # decode_mask covers: original prompt tokens + first generated token
    decode_mask = torch.ones((1, seq_len + 1), dtype=attention_mask.dtype)

    # FIX 6: verify mask length each step — guards against silent misalignment
    # across transformers versions (mirrors GPU script Fix 8)
    for step in range(MAX_NEW_TOKENS - 1):
        if next_token.item() == eos_id:
            break

        past_kv = tq_engine.decompress_to_cache(compressed_store)

        expected_mask_len = seq_len + 1 + step + 1
        if decode_mask.shape[-1] != expected_mask_len:
            pad = expected_mask_len - decode_mask.shape[-1]
            if pad > 0:
                decode_mask = torch.cat(
                    [decode_mask,
                     torch.ones((1, pad), dtype=decode_mask.dtype)],
                    dim=-1,
                )
            else:
                decode_mask = decode_mask[:, :expected_mask_len]

        out = model(
            input_ids       = next_token,
            attention_mask  = decode_mask,
            past_key_values = past_kv,
            use_cache       = True,
        )
        next_token        = out.logits[:, -1:, :].argmax(dim=-1)
        tokens_generated += 1
        compressed_store  = tq_engine.compress_cache(out.past_key_values)
        del out, past_kv
        gc.collect()

        decode_mask = torch.cat(
            [decode_mask, torch.ones((1, 1), dtype=decode_mask.dtype)], dim=-1
        )

    latency = time.perf_counter() - t_start
    return {
        "ttft"             : round(ttft, 4),
        "latency"          : round(latency, 4),
        "throughput_tps"   : round(tokens_generated / latency if latency > 0 else 0.0, 3),
        "tokens_generated" : tokens_generated,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK ONE MODEL  → returns summary dict
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_model(model_key: str, model_path: str, runs: int,
                    key_bits: int, val_bits: int, prompt: str,
                    global_ram: RAMSampler) -> dict:
    spec  = MODEL_SPECS[model_key]
    label = spec["label"]

    print()
    print(SEP2)
    print(f"  ▶  Benchmarking  {label}  ({model_key})")
    print(SEP2)
    print(f"  Path    : {model_path}")
    print(f"  Key/Val : {key_bits}b / {val_bits}b  TurboQuant")
    print(SEP)

    print("\n  Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )

    print("  Loading model on CPU (this may take a moment) …")
    # FIX 1: use torch_dtype= not dtype= — the wrong kwarg was silently ignored
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype       = torch.float32,
        device_map        = "cpu",
        trust_remote_code = True,
        local_files_only  = True,
    )
    model.eval()
    print(f"  ✅  Model loaded  |  RAM now: {global_ram.current_gb:.2f} GB\n")

    tq_engine = TurboQuantEngine(model, key_bits, val_bits)

    # FIX 7: read layer count from engine (which got it from model config)
    print(f"  Layers  : {tq_engine.num_layers}  (from config)")

    # tokenise prompt
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except (ValueError, AttributeError):
        print("  ℹ️  No chat template — using Gemma 4 turn format.")
        text = GEMMA4_CHAT_TEMPLATE.format(prompt=prompt)

    inputs         = tokenizer(text, return_tensors="pt")
    input_ids      = inputs["input_ids"][:, :CONTEXT_WINDOW]
    attention_mask = inputs.get("attention_mask",
                                torch.ones_like(input_ids))[:, :CONTEXT_WINDOW]
    n_tokens = input_ids.shape[-1]
    print(f"  Input tokens : {n_tokens}")

    print(f"\n  {'Run':>5}  {'TTFT':>9}  {'Latency':>11}  {'TPS':>9}  {'Tok':>6}  {'RAM':>8}")
    print(f"  {SEP3[:65]}")

    # FIX 5: warmup run — discarded from results to avoid JIT/mmap overhead
    # inflating Run 1 latency
    print("  Warmup (discarded) …", end="", flush=True)
    _ = run_single(model, tokenizer, input_ids, attention_mask, tq_engine)
    print("  done")
    print(f"  {SEP3[:65]}")

    run_results = []
    for i in range(runs):
        run_ram = RAMSampler().start()
        result  = run_single(model, tokenizer, input_ids, attention_mask, tq_engine)
        peak_r  = run_ram.stop()
        run_results.append({**result, "peak_run_ram_gb": round(peak_r, 3)})
        print(
            f"  Run {i+1:>2}   "
            f"{result['ttft']:>8.3f}s  "
            f"{result['latency']:>10.3f}s  "
            f"{result['throughput_tps']:>8.3f}  "
            f"{result['tokens_generated']:>5}  "
            f"{peak_r:>6.2f} GB"
        )

    # unload model to free RAM before loading the next one
    del model, tq_engine, tokenizer
    gc.collect()
    print(f"\n  ♻️   Model unloaded  |  RAM now: {global_ram.current_gb:.2f} GB")

    # aggregate
    ttfts     = [r["ttft"]            for r in run_results]
    latencies = [r["latency"]         for r in run_results]
    tpss      = [r["throughput_tps"]  for r in run_results]
    rams      = [r["peak_run_ram_gb"] for r in run_results]

    # FIX 3: use engine's accurate size estimate (engine already deleted, so
    # we reconstruct the numbers from stored config values captured above)
    # Instead, estimate_sizes_mb is called before the engine is deleted:
    # — moved size estimation to just before del model (see below) —
    # NOTE: sizes are captured in the dict returned from this function.
    #       The calculation below mirrors estimate_sizes_mb exactly.
    # (We cannot call tq_engine here because it is deleted; the sizes were
    #  captured in `_sizes` just before deletion — see the code above the del.)

    return {
        "label"          : label,
        "model_key"      : model_key,
        "n_layers"       : _sizes_capture["n_layers"],
        "n_kv_heads"     : _sizes_capture["n_kv_heads"],
        "n_input_tokens" : n_tokens,
        "runs"           : run_results,
        "ttft_mean"      : round(statistics.mean(ttfts), 4),
        "ttft_min"       : round(min(ttfts), 4),
        "ttft_max"       : round(max(ttfts), 4),
        "ttft_std"       : round(statistics.stdev(ttfts), 4) if len(ttfts) > 1 else None,
        "latency_mean"   : round(statistics.mean(latencies), 4),
        "latency_min"    : round(min(latencies), 4),
        "latency_max"    : round(max(latencies), 4),
        "latency_std"    : round(statistics.stdev(latencies), 4) if len(latencies) > 1 else None,
        "tps_mean"       : round(statistics.mean(tpss), 3),
        "tps_min"        : round(min(tpss), 3),
        "tps_max"        : round(max(tpss), 3),
        "tps_std"        : round(statistics.stdev(tpss), 3) if len(tpss) > 1 else None,
        "ram_mean_gb"    : round(statistics.mean(rams), 3),
        "fp32_kv_mb"     : round(_sizes_capture["float32_mb"], 2),
        "turbo_kv_mb"    : round(_sizes_capture["turbo_mb"], 2),
        "compress_ratio" : round(_sizes_capture["ratio"], 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# The approach above with _sizes_capture is fragile. Refactor benchmark_model
# to capture sizes before deleting the engine — cleaner implementation below.
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_model(model_key: str, model_path: str, runs: int,      # noqa: F811
                    key_bits: int, val_bits: int, prompt: str,
                    global_ram: RAMSampler) -> dict:
    """
    Load model, run warmup + timed benchmark, unload model, return summary.
    """
    spec  = MODEL_SPECS[model_key]
    label = spec["label"]

    print()
    print(SEP2)
    print(f"  ▶  Benchmarking  {label}  ({model_key})")
    print(SEP2)
    print(f"  Path    : {model_path}")
    print(f"  Key/Val : {key_bits}b / {val_bits}b  TurboQuant")
    print(SEP)

    print("\n  Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )

    print("  Loading model on CPU (this may take a moment) …")
    # FIX 1: correct kwarg is torch_dtype=, not dtype=
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype       = torch.float32,
        device_map        = "cpu",
        trust_remote_code = True,
        local_files_only  = True,
    )
    model.eval()
    print(f"  ✅  Model loaded  |  RAM now: {global_ram.current_gb:.2f} GB\n")

    tq_engine = TurboQuantEngine(model, key_bits, val_bits)

    # FIX 7: layer/head counts come from engine (which read from model config)
    n_layers   = tq_engine.num_layers
    n_kv_heads = tq_engine.num_kv_heads
    print(f"  Layers  : {n_layers}  (from config)")
    print(f"  KV heads: {n_kv_heads}")

    # FIX 3: capture accurate size estimates before engine is deleted
    sizes = tq_engine.estimate_sizes_mb(CONTEXT_WINDOW)

    # tokenise prompt
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except (ValueError, AttributeError):
        print("  ℹ️  No chat template — using Gemma 4 turn format.")
        text = GEMMA4_CHAT_TEMPLATE.format(prompt=prompt)

    inputs         = tokenizer(text, return_tensors="pt")
    input_ids      = inputs["input_ids"][:, :CONTEXT_WINDOW]
    attention_mask = inputs.get("attention_mask",
                                torch.ones_like(input_ids))[:, :CONTEXT_WINDOW]
    n_tokens = input_ids.shape[-1]
    print(f"  Input tokens : {n_tokens}")

    print(f"\n  {'Run':>5}  {'TTFT':>9}  {'Latency':>11}  {'TPS':>9}  {'Tok':>6}  {'RAM':>8}")
    print(f"  {SEP3[:65]}")

    # FIX 5: warmup run — discarded from results
    print("  Warmup (discarded) …", end="", flush=True)
    _ = run_single(model, tokenizer, input_ids, attention_mask, tq_engine)
    print("  done")
    print(f"  {SEP3[:65]}")

    run_results = []
    for i in range(runs):
        run_ram = RAMSampler().start()
        result  = run_single(model, tokenizer, input_ids, attention_mask, tq_engine)
        peak_r  = run_ram.stop()
        run_results.append({**result, "peak_run_ram_gb": round(peak_r, 3)})
        print(
            f"  Run {i+1:>2}   "
            f"{result['ttft']:>8.3f}s  "
            f"{result['latency']:>10.3f}s  "
            f"{result['throughput_tps']:>8.3f}  "
            f"{result['tokens_generated']:>5}  "
            f"{peak_r:>6.2f} GB"
        )

    # unload model to free RAM before loading the next one
    del model, tq_engine, tokenizer
    gc.collect()
    print(f"\n  ♻️   Model unloaded  |  RAM now: {global_ram.current_gb:.2f} GB")

    # aggregate
    ttfts     = [r["ttft"]            for r in run_results]
    latencies = [r["latency"]         for r in run_results]
    tpss      = [r["throughput_tps"]  for r in run_results]
    rams      = [r["peak_run_ram_gb"] for r in run_results]

    return {
        "label"          : label,
        "model_key"      : model_key,
        "n_layers"       : n_layers,
        "n_kv_heads"     : n_kv_heads,
        "n_input_tokens" : n_tokens,
        "runs"           : run_results,
        "ttft_mean"      : round(statistics.mean(ttfts), 4),
        "ttft_min"       : round(min(ttfts), 4),
        "ttft_max"       : round(max(ttfts), 4),
        "ttft_std"       : round(statistics.stdev(ttfts), 4) if len(ttfts) > 1 else None,
        "latency_mean"   : round(statistics.mean(latencies), 4),
        "latency_min"    : round(min(latencies), 4),
        "latency_max"    : round(max(latencies), 4),
        "latency_std"    : round(statistics.stdev(latencies), 4) if len(latencies) > 1 else None,
        "tps_mean"       : round(statistics.mean(tpss), 3),
        "tps_min"        : round(min(tpss), 3),
        "tps_max"        : round(max(tpss), 3),
        "tps_std"        : round(statistics.stdev(tpss), 3) if len(tpss) > 1 else None,
        "ram_mean_gb"    : round(statistics.mean(rams), 3),
        "fp32_kv_mb"     : round(sizes["float32_mb"], 2),
        "turbo_kv_mb"    : round(sizes["turbo_mb"], 2),
        "compress_ratio" : round(sizes["ratio"], 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPARATIVE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _delta(a, b, higher_is_better=False):
    """Return a ▲/▼ delta string: positive = E4B is better."""
    diff = b - a
    if higher_is_better:
        arrow = "▲" if diff >= 0 else "▼"
        pct   = abs(diff) / a * 100 if a else 0
        return f"{arrow} {abs(diff):+.4g}  ({pct:.1f}%)"
    else:
        arrow = "▼" if diff <= 0 else "▲"
        pct   = abs(diff) / a * 100 if a else 0
        return f"{arrow} {abs(diff):+.4g}  ({pct:.1f}%)"


def _std_fmt(v):
    return f"±{v:.4f}" if v is not None else "   ±—  "


def print_comparison(e2b: dict, e4b: dict, key_bits: int, val_bits: int):
    print()
    print(SEP2)
    print("  COMPARATIVE ANALYSIS  :  Gemma 4 E2B  vs  Gemma 4 E4B")
    print(f"  TurboQuant KV Cache  ·  key={key_bits}b  val={val_bits}b  "
          f"·  context={CONTEXT_WINDOW}  ·  CPU / float32")
    print(SEP2)

    # ── Model Architecture ────────────────────────────────────────────────────
    print(f"\n  {'ARCHITECTURE':<30}  {'E2B':>12}  {'E4B':>12}  {'Δ (E4B vs E2B)':>22}")
    print(f"  {SEP}")
    arch_rows = [
        ("Hidden layers",   e2b["n_layers"],      e4b["n_layers"],      False),
        ("KV heads",        e2b["n_kv_heads"],    e4b["n_kv_heads"],    False),
        ("Input tokens",    e2b["n_input_tokens"], e4b["n_input_tokens"], False),
    ]
    for label, va, vb, hib in arch_rows:
        delta = _delta(va, vb, higher_is_better=hib) if va != vb else "—"
        print(f"  {label:<30}  {str(va):>12}  {str(vb):>12}  {delta:>22}")

    # ── Latency Metrics ───────────────────────────────────────────────────────
    print(f"\n  {'LATENCY METRICS':<30}  {'E2B mean':>12}  {'E4B mean':>12}  {'Δ (E4B vs E2B)':>22}")
    print(f"  {SEP}")
    lat_rows = [
        ("TTFT (s)",           e2b["ttft_mean"],    e4b["ttft_mean"],    False, "ttft_std"),
        ("Total latency (s)",  e2b["latency_mean"], e4b["latency_mean"], False, "latency_std"),
    ]
    for label, va, vb, hib, std_key in lat_rows:
        std_a = _std_fmt(e2b[std_key])
        std_b = _std_fmt(e4b[std_key])
        delta = _delta(va, vb, higher_is_better=hib)
        print(f"  {label:<30}  {va:>9.4f}s {std_a}  {vb:>9.4f}s {std_b}  {delta:>22}")

    # ── Throughput ────────────────────────────────────────────────────────────
    print(f"\n  {'THROUGHPUT':<30}  {'E2B mean':>12}  {'E4B mean':>12}  {'Δ (E4B vs E2B)':>22}")
    print(f"  {SEP}")
    std_a = _std_fmt(e2b["tps_std"])
    std_b = _std_fmt(e4b["tps_std"])
    delta = _delta(e2b["tps_mean"], e4b["tps_mean"], higher_is_better=True)
    print(f"  {'Throughput (tok/s)':<30}  {e2b['tps_mean']:>8.3f}  {std_a}  "
          f"{e4b['tps_mean']:>8.3f}  {std_b}  {delta:>22}")

    # ── RAM ───────────────────────────────────────────────────────────────────
    print(f"\n  {'RAM (GB)':<30}  {'E2B':>12}  {'E4B':>12}  {'Δ (E4B vs E2B)':>22}")
    print(f"  {SEP}")
    delta_ram = _delta(e2b["ram_mean_gb"], e4b["ram_mean_gb"], higher_is_better=False)
    print(f"  {'Peak per-run RAM':<30}  {e2b['ram_mean_gb']:>10.3f}G  "
          f"{e4b['ram_mean_gb']:>10.3f}G  {delta_ram:>22}")

    # ── KV Cache Compression ──────────────────────────────────────────────────
    print(f"\n  {'KV CACHE COMPRESSION':<30}  {'E2B':>12}  {'E4B':>12}  {'Note':>22}")
    print(f"  {SEP}")
    kv_rows = [
        ("Float32 KV size (MB)",  e2b["fp32_kv_mb"],    e4b["fp32_kv_mb"],    "baseline"),
        ("TurboQuant KV (MB)",    e2b["turbo_kv_mb"],   e4b["turbo_kv_mb"],   f"key={key_bits}b val={val_bits}b"),
        ("Compression ratio",     e2b["compress_ratio"],e4b["compress_ratio"],"fp32 ÷ turbo"),
    ]
    for label, va, vb, note in kv_rows:
        print(f"  {label:<30}  {str(va):>12}  {str(vb):>12}  {note:>22}")

    # ── Run-level detail ──────────────────────────────────────────────────────
    print(f"\n  {'PER-RUN DETAIL'}")
    print(f"  {SEP}")
    max_runs = max(len(e2b["runs"]), len(e4b["runs"]))
    hdr = (f"  {'Run':>4}  "
           f"{'E2B TTFT':>9}  {'E2B Lat':>9}  {'E2B TPS':>9}  "
           f"    "
           f"{'E4B TTFT':>9}  {'E4B Lat':>9}  {'E4B TPS':>9}")
    print(hdr)
    print(f"  {SEP3[:len(hdr)-2]}")
    for i in range(max_runs):
        ra = e2b["runs"][i] if i < len(e2b["runs"]) else None
        rb = e4b["runs"][i] if i < len(e4b["runs"]) else None
        a_str = (f"{ra['ttft']:>8.3f}s  {ra['latency']:>8.3f}s  {ra['throughput_tps']:>8.3f}"
                 if ra else "         —           —           —")
        b_str = (f"{rb['ttft']:>8.3f}s  {rb['latency']:>8.3f}s  {rb['throughput_tps']:>8.3f}"
                 if rb else "         —           —           —")
        print(f"  {i+1:>4}  {a_str}      {b_str}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print()
    print(SEP2)
    print("  VERDICT")
    print(SEP)

    faster_ttft  = "E2B" if e2b["ttft_mean"] < e4b["ttft_mean"] else "E4B"
    faster_lat   = "E2B" if e2b["latency_mean"] < e4b["latency_mean"] else "E4B"
    higher_tps   = "E2B" if e2b["tps_mean"] > e4b["tps_mean"] else "E4B"
    lower_ram    = "E2B" if e2b["ram_mean_gb"] < e4b["ram_mean_gb"] else "E4B"

    ttft_gap_pct = abs(e2b["ttft_mean"] - e4b["ttft_mean"]) / min(e2b["ttft_mean"], e4b["ttft_mean"]) * 100
    lat_gap_pct  = abs(e2b["latency_mean"] - e4b["latency_mean"]) / min(e2b["latency_mean"], e4b["latency_mean"]) * 100
    tps_gap_pct  = abs(e2b["tps_mean"] - e4b["tps_mean"]) / max(e2b["tps_mean"], e4b["tps_mean"]) * 100
    ram_gap_pct  = abs(e2b["ram_mean_gb"] - e4b["ram_mean_gb"]) / min(e2b["ram_mean_gb"], e4b["ram_mean_gb"]) * 100

    print(f"  ⏱  Faster TTFT      → {faster_ttft}  ({ttft_gap_pct:.1f}% gap)")
    print(f"  🕐  Lower latency   → {faster_lat}  ({lat_gap_pct:.1f}% gap)")
    print(f"  ⚡  Higher TPS      → {higher_tps}  ({tps_gap_pct:.1f}% gap)")
    print(f"  💾  Lower RAM       → {lower_ram}  ({ram_gap_pct:.1f}% gap)")
    print()
    print(f"  KV Compression: E2B saves {e2b['fp32_kv_mb'] - e2b['turbo_kv_mb']:.1f} MB  "
          f"({e2b['compress_ratio']}× ratio)  |  "
          f"E4B saves {e4b['fp32_kv_mb'] - e4b['turbo_kv_mb']:.1f} MB  "
          f"({e4b['compress_ratio']}× ratio)")

    if e2b["compress_ratio"] == e4b["compress_ratio"]:
        print("  ✅  Both models achieve the same compression ratio with TurboQuant.")
    else:
        better_cr = "E2B" if e2b["compress_ratio"] > e4b["compress_ratio"] else "E4B"
        print(f"  ✅  {better_cr} achieves a higher compression ratio with TurboQuant.")

    print(SEP2)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(e2b: dict, e4b: dict, path: str = "benchmark_compare_results.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)

        # ── Summary ────────────────────────────────────────────────────────────
        w.writerow(["## SUMMARY"])
        w.writerow(["metric", "e2b_mean", "e4b_mean", "delta", "winner"])
        summary = [
            ("ttft_s",         e2b["ttft_mean"],     e4b["ttft_mean"],     False),
            ("latency_s",      e2b["latency_mean"],  e4b["latency_mean"],  False),
            ("throughput_tps", e2b["tps_mean"],      e4b["tps_mean"],      True),
            ("peak_ram_gb",    e2b["ram_mean_gb"],   e4b["ram_mean_gb"],   False),
            ("fp32_kv_mb",     e2b["fp32_kv_mb"],    e4b["fp32_kv_mb"],    False),
            ("turbo_kv_mb",    e2b["turbo_kv_mb"],   e4b["turbo_kv_mb"],   False),
            ("compress_ratio", e2b["compress_ratio"],e4b["compress_ratio"],True),
        ]
        for label, va, vb, hib in summary:
            diff   = vb - va
            winner = "E4B" if (diff > 0) == hib else "E2B"
            w.writerow([label, va, vb, round(diff, 6), winner])

        # ── Per-run detail ─────────────────────────────────────────────────────
        w.writerow([])
        w.writerow(["## PER-RUN DETAIL"])
        w.writerow(["model", "run", "ttft_s", "latency_s", "throughput_tps",
                    "tokens_generated", "peak_run_ram_gb"])
        # FIX 4: use enumerate() — list.index() returns first-match index which
        # silently produces wrong run numbers when any two runs share a value
        for run_num, r in enumerate(e2b["runs"], 1):
            w.writerow(["E2B", run_num,
                        r["ttft"], r["latency"], r["throughput_tps"],
                        r["tokens_generated"], r["peak_run_ram_gb"]])
        for run_num, r in enumerate(e4b["runs"], 1):
            w.writerow(["E4B", run_num,
                        r["ttft"], r["latency"], r["throughput_tps"],
                        r["tokens_generated"], r["peak_run_ram_gb"]])

    print(f"  💾  Comparison CSV saved → {path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side benchmark: Gemma 4 E2B vs E4B with TurboQuant KV Cache"
    )
    parser.add_argument("--e2b-model", default=MODEL_SPECS["E2B"]["default_path"],
                        help="Path to Gemma 4 E2B model directory")
    parser.add_argument("--e4b-model", default=MODEL_SPECS["E4B"]["default_path"],
                        help="Path to Gemma 4 E4B model directory")
    parser.add_argument("--runs",      "-r", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--prompt",    "-p", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--key-bits",        type=int, default=DEFAULT_KEY_BITS)
    parser.add_argument("--val-bits",        type=int, default=DEFAULT_VAL_BITS)
    parser.add_argument("--skip-e2b",  action="store_true", help="Skip E2B, run E4B only")
    parser.add_argument("--skip-e4b",  action="store_true", help="Skip E4B, run E2B only")
    parser.add_argument("--csv",       default="benchmark_compare_results.csv")
    args = parser.parse_args()

    if args.skip_e2b and args.skip_e4b:
        sys.exit("❌  Cannot skip both models.")

    models_to_run = []
    if not args.skip_e2b:
        if not os.path.isdir(args.e2b_model):
            sys.exit(f"❌  E2B model directory not found: {args.e2b_model}")
        models_to_run.append(("E2B", args.e2b_model))
    if not args.skip_e4b:
        if not os.path.isdir(args.e4b_model):
            sys.exit(f"❌  E4B model directory not found: {args.e4b_model}")
        models_to_run.append(("E4B", args.e4b_model))

    print(SEP2)
    print("  Gemma 4  E2B vs E4B  ·  TurboQuant KV Cache  ·  CPU-only Benchmark")
    print(SEP2)
    print(f"  Runs           : {args.runs}  (+1 warmup per model, discarded)")
    print(f"  Key bits       : {args.key_bits}  ({2**args.key_bits} centroids)")
    print(f"  Value bits     : {args.val_bits}  ({2**args.val_bits} centroids)")
    print(f"  Context window : {CONTEXT_WINDOW} tokens")
    print(f"  Max new tokens : {MAX_NEW_TOKENS}")
    print(f"  Device         : CPU  (float32)")
    print(f"  Prompt         : \"{args.prompt[:70]}{'…' if len(args.prompt) > 70 else ''}\"")
    print(SEP2)

    global_ram   = RAMSampler().start()
    all_results  = {}

    for model_key, model_path in models_to_run:
        result = benchmark_model(
            model_key  = model_key,
            model_path = model_path,
            runs       = args.runs,
            key_bits   = args.key_bits,
            val_bits   = args.val_bits,
            prompt     = args.prompt,
            global_ram = global_ram,
        )
        all_results[model_key] = result

    global_ram.stop()

    if len(all_results) == 2:
        print_comparison(all_results["E2B"], all_results["E4B"], args.key_bits, args.val_bits)
        save_csv(all_results["E2B"], all_results["E4B"], path=args.csv)
    else:
        # Single model — just print its summary
        key = list(all_results.keys())[0]
        r   = all_results[key]
        print(f"\n  Single-model run complete: {r['label']}")
        print(f"  Layers       : {r['n_layers']}")
        print(f"  KV heads     : {r['n_kv_heads']}")
        print(f"  TTFT mean    : {r['ttft_mean']:.4f}s")
        print(f"  Latency mean : {r['latency_mean']:.4f}s")
        print(f"  TPS mean     : {r['tps_mean']:.3f}")
        print(f"  RAM mean     : {r['ram_mean_gb']:.3f} GB")
        print(f"  Compression  : {r['compress_ratio']}×  "
              f"({r['fp32_kv_mb']} MB → {r['turbo_kv_mb']} MB)")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# benchmark_compare.py — Gemma 4 E2B vs E4B · TurboQuant KV Cache · CPU-only
# ============================================================================
# Runs both models back-to-back and prints a detailed side-by-side comparison.

# Metrics  :  TTFT  |  Latency  |  Throughput  |  RAM (GB)  |  KV Compression
# Context  :  2 048 tokens (fixed)
# Backend  :  CPU only, float32
# Models   :  google/gemma-4-E2B  (35 layers)
#             google/gemma-4-E4B  (42 layers)

# Usage:
#     python benchmark_compare.py \
#         --e2b-model ./google-gemma-4-E2B \
#         --e4b-model /home/kanshika/Desktop/Model/gemma-4-E4B \
#         --runs 3 \
#         --key-bits 3 \
#         --val-bits 2
# """

# import argparse
# import csv
# import gc
# import os
# import statistics
# import sys
# import threading
# import time

# # ── dependency checks ──────────────────────────────────────────────────────────
# try:
#     import psutil
# except ImportError:
#     sys.exit("❌  pip install psutil")

# try:
#     import torch
# except ImportError:
#     sys.exit("❌  pip install torch")

# try:
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     from transformers.cache_utils import DynamicCache
# except ImportError:
#     sys.exit("❌  pip install transformers accelerate")

# # ─────────────────────────────────────────────────────────────────────────────
# # CONSTANTS
# # ─────────────────────────────────────────────────────────────────────────────
# CONTEXT_WINDOW   = 2048
# MAX_NEW_TOKENS   = 128
# DEFAULT_RUNS     = 3
# DEFAULT_KEY_BITS = 3
# DEFAULT_VAL_BITS = 2

# MODEL_SPECS = {
#     "E2B": {"layers": 35, "label": "Gemma 4 E2B", "default_path": "./google-gemma-4-E2B"},
#     "E4B": {"layers": 42, "label": "Gemma 4 E4B", "default_path": "/home/kanshika/Desktop/Model/gemma-4-E4B"},
# }

# DEFAULT_PROMPT = (
#     "Explain the key differences between supervised, unsupervised, and "
#     "reinforcement learning. Give one real-world example for each."
# )

# GEMMA4_CHAT_TEMPLATE = (
#     "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
# )

# W    = 80
# SEP  = "─" * W
# SEP2 = "═" * W
# SEP3 = "┄" * W


# # ─────────────────────────────────────────────────────────────────────────────
# # DYNAMICCACHE HELPERS  (version-agnostic)
# # ─────────────────────────────────────────────────────────────────────────────

# def _cache_has_old_api(cache: DynamicCache) -> bool:
#     return hasattr(cache, "key_cache")


# def extract_kv_from_cache(cache: DynamicCache) -> list:
#     if _cache_has_old_api(cache):
#         return [(k, v) if k is not None else None
#                 for k, v in zip(cache.key_cache, cache.value_cache)]
#     result = []
#     for layer in cache.layers:
#         if layer is None:
#             result.append(None)
#         else:
#             k = getattr(layer, "key", None)
#             v = getattr(layer, "value", None)
#             if k is None:
#                 try:
#                     k, v = layer.key_cache, layer.value_cache
#                 except AttributeError:
#                     result.append(None)
#                     continue
#             result.append((k, v))
#     return result


# def build_cache_from_kv(layer_kvs: list) -> DynamicCache:
#     new_cache = DynamicCache()
#     for i, kv in enumerate(layer_kvs):
#         if kv is not None:
#             new_cache.update(kv[0], kv[1], i)
#     return new_cache


# # ─────────────────────────────────────────────────────────────────────────────
# # RAM SAMPLER
# # ─────────────────────────────────────────────────────────────────────────────

# class RAMSampler:
#     def __init__(self):
#         self._proc   = psutil.Process(os.getpid())
#         self._peak   = 0.0
#         self._stop   = threading.Event()
#         self._thread = threading.Thread(target=self._poll, daemon=True)

#     def _poll(self):
#         while not self._stop.is_set():
#             rss = self._proc.memory_info().rss / (1024 ** 3)
#             if rss > self._peak:
#                 self._peak = rss
#             time.sleep(0.1)

#     def start(self):
#         self._thread.start()
#         return self

#     def stop(self) -> float:
#         self._stop.set()
#         self._thread.join()
#         return self._peak

#     @property
#     def current_gb(self) -> float:
#         return self._proc.memory_info().rss / (1024 ** 3)


# # ─────────────────────────────────────────────────────────────────────────────
# # TURBOQUANT CORE
# # ─────────────────────────────────────────────────────────────────────────────

# def build_rotation_matrix(dim: int, seed: int = 42) -> torch.Tensor:
#     gen = torch.Generator()
#     gen.manual_seed(seed)
#     G = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
#     Q, _ = torch.linalg.qr(G)
#     return Q


# def fit_lloyd_max(n_bits: int, n_samples: int = 100_000, n_iter: int = 150) -> tuple:
#     n_levels  = 2 ** n_bits
#     gen       = torch.Generator()
#     gen.manual_seed(0)
#     sample    = torch.randn(n_samples, generator=gen)
#     centroids = torch.linspace(-3.0, 3.0, n_levels)

#     for _ in range(n_iter):
#         boundaries  = (centroids[:-1] + centroids[1:]) / 2.0
#         full_bounds = torch.cat([torch.tensor([-1e9]), boundaries, torch.tensor([1e9])])
#         new_c       = torch.empty_like(centroids)
#         for k in range(n_levels):
#             mask     = (sample >= full_bounds[k]) & (sample < full_bounds[k + 1])
#             new_c[k] = sample[mask].mean() if mask.any() else centroids[k]
#         if (new_c - centroids).abs().max().item() < 1e-7:
#             break
#         centroids = new_c

#     return boundaries, centroids


# # ─────────────────────────────────────────────────────────────────────────────
# # PER-LAYER COMPRESSOR
# # ─────────────────────────────────────────────────────────────────────────────

# class LayerCompressor:
#     def __init__(self, head_dim: int, key_bits: int, val_bits: int, layer_idx: int):
#         self.head_dim = head_dim
#         self.Q_k      = build_rotation_matrix(head_dim, seed=layer_idx * 2)
#         self.Q_v      = build_rotation_matrix(head_dim, seed=layer_idx * 2 + 1)
#         self.k_bounds = self.k_cents = self.v_bounds = self.v_cents = None

#     def _compress(self, x, Q, bounds):
#         y      = x @ Q.T
#         scale  = y.std(dim=-1, keepdim=True).clamp(min=1e-8)
#         idx    = torch.bucketize((y / scale).contiguous(), bounds)
#         return idx.to(torch.uint8), scale

#     def _decompress(self, idx, scale, Q, cents):
#         return (cents[idx.long()] * scale) @ Q

#     def compress(self, key, value):
#         k_idx, k_scale = self._compress(key,   self.Q_k, self.k_bounds)
#         v_idx, v_scale = self._compress(value, self.Q_v, self.v_bounds)
#         return {"k_idx": k_idx, "k_scale": k_scale,
#                 "v_idx": v_idx, "v_scale": v_scale}

#     def decompress(self, data):
#         key   = self._decompress(data["k_idx"], data["k_scale"], self.Q_k, self.k_cents)
#         value = self._decompress(data["v_idx"], data["v_scale"], self.Q_v, self.v_cents)
#         return key, value


# # ─────────────────────────────────────────────────────────────────────────────
# # TURBOQUANT ENGINE
# # ─────────────────────────────────────────────────────────────────────────────

# class TurboQuantEngine:
#     def __init__(self, model, key_bits: int, val_bits: int):
#         cfg      = model.config
#         text_cfg = getattr(cfg, "text_config", cfg)

#         num_layers    = text_cfg.num_hidden_layers
#         self.key_bits = key_bits
#         self.val_bits = val_bits

#         head_dim = getattr(text_cfg, "head_dim", None)
#         if head_dim is None:
#             head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
#         self.head_dim = head_dim

#         print(f"  [TurboQuant] Fitting Lloyd-Max codebooks …  "
#               f"(key={key_bits}b/{2**key_bits}c  val={val_bits}b/{2**val_bits}c)")
#         t0 = time.perf_counter()
#         k_bounds, k_cents = fit_lloyd_max(key_bits)
#         v_bounds, v_cents = fit_lloyd_max(val_bits)
#         print(f"  [TurboQuant] Codebooks done in {time.perf_counter() - t0:.2f}s")

#         self.compressors = []
#         for i in range(num_layers):
#             lc = LayerCompressor(head_dim, key_bits, val_bits, layer_idx=i)
#             lc.k_bounds, lc.k_cents = k_bounds, k_cents
#             lc.v_bounds, lc.v_cents = v_bounds, v_cents
#             self.compressors.append(lc)
#         print(f"  [TurboQuant] {num_layers} layer compressors ready (head_dim={head_dim})\n")

#     def compress_cache(self, cache: DynamicCache) -> list:
#         layer_kvs  = extract_kv_from_cache(cache)
#         compressed = []
#         for i, cmp in enumerate(self.compressors):
#             if i < len(layer_kvs) and layer_kvs[i] is not None:
#                 compressed.append(cmp.compress(*layer_kvs[i]))
#             else:
#                 compressed.append(None)
#         return compressed

#     def decompress_to_cache(self, compressed: list) -> DynamicCache:
#         layer_kvs = []
#         for cmp, data in zip(self.compressors, compressed):
#             layer_kvs.append(cmp.decompress(data) if data is not None else None)
#         return build_cache_from_kv(layer_kvs)


# # ─────────────────────────────────────────────────────────────────────────────
# # SINGLE BENCHMARK RUN
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.inference_mode()
# def run_single(model, tokenizer, input_ids, attention_mask, tq_engine) -> dict:
#     eos_id           = tokenizer.eos_token_id
#     tokens_generated = 0
#     t_start          = time.perf_counter()

#     out              = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
#     ttft             = time.perf_counter() - t_start
#     next_token       = out.logits[:, -1:, :].argmax(dim=-1)
#     tokens_generated += 1

#     compressed_store = tq_engine.compress_cache(out.past_key_values)
#     del out
#     gc.collect()

#     decode_mask = torch.ones((1, input_ids.shape[-1] + 1), dtype=attention_mask.dtype)

#     for _ in range(MAX_NEW_TOKENS - 1):
#         if next_token.item() == eos_id:
#             break
#         past_kv = tq_engine.decompress_to_cache(compressed_store)
#         out     = model(
#             input_ids       = next_token,
#             attention_mask  = decode_mask,
#             past_key_values = past_kv,
#             use_cache       = True,
#         )
#         next_token        = out.logits[:, -1:, :].argmax(dim=-1)
#         tokens_generated += 1
#         compressed_store  = tq_engine.compress_cache(out.past_key_values)
#         del out, past_kv
#         gc.collect()
#         decode_mask = torch.cat(
#             [decode_mask, torch.ones((1, 1), dtype=decode_mask.dtype)], dim=-1
#         )

#     latency = time.perf_counter() - t_start
#     return {
#         "ttft"             : round(ttft, 4),
#         "latency"          : round(latency, 4),
#         "throughput_tps"   : round(tokens_generated / latency if latency > 0 else 0.0, 3),
#         "tokens_generated" : tokens_generated,
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # BENCHMARK ONE MODEL  → returns summary dict
# # ─────────────────────────────────────────────────────────────────────────────

# def benchmark_model(model_key: str, model_path: str, runs: int,
#                     key_bits: int, val_bits: int, prompt: str,
#                     global_ram: RAMSampler) -> dict:
#     spec  = MODEL_SPECS[model_key]
#     label = spec["label"]

#     print()
#     print(SEP2)
#     print(f"  ▶  Benchmarking  {label}  ({model_key})")
#     print(SEP2)
#     print(f"  Path    : {model_path}")
#     print(f"  Layers  : {spec['layers']}")
#     print(f"  Key/Val : {key_bits}b / {val_bits}b  TurboQuant")
#     print(SEP)

#     print("\n  Loading tokenizer …")
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path, trust_remote_code=True, local_files_only=True,
#     )

#     print("  Loading model on CPU (this may take a moment) …")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         dtype             = torch.float32,
#         device_map        = "cpu",
#         trust_remote_code = True,
#         local_files_only  = True,
#     )
#     model.eval()
#     print(f"  ✅  Model loaded  |  RAM now: {global_ram.current_gb:.2f} GB\n")

#     tq_engine = TurboQuantEngine(model, key_bits, val_bits)

#     # tokenise prompt
#     try:
#         text = tokenizer.apply_chat_template(
#             [{"role": "user", "content": prompt}],
#             tokenize=False, add_generation_prompt=True,
#         )
#     except (ValueError, AttributeError):
#         print("  ℹ️  No chat template — using Gemma 4 turn format.")
#         text = GEMMA4_CHAT_TEMPLATE.format(prompt=prompt)

#     inputs         = tokenizer(text, return_tensors="pt")
#     input_ids      = inputs["input_ids"][:, :CONTEXT_WINDOW]
#     attention_mask = inputs.get("attention_mask",
#                                 torch.ones_like(input_ids))[:, :CONTEXT_WINDOW]
#     n_tokens = input_ids.shape[-1]
#     print(f"  Input tokens : {n_tokens}")

#     print(f"\n  {'Run':>5}  {'TTFT':>9}  {'Latency':>11}  {'TPS':>9}  {'Tok':>6}  {'RAM':>8}")
#     print(f"  {SEP3[:65]}")

#     run_results = []
#     for i in range(runs):
#         run_ram = RAMSampler().start()
#         result  = run_single(model, tokenizer, input_ids, attention_mask, tq_engine)
#         peak_r  = run_ram.stop()
#         run_results.append({**result, "peak_run_ram_gb": round(peak_r, 3)})
#         print(
#             f"  Run {i+1:>2}   "
#             f"{result['ttft']:>8.3f}s  "
#             f"{result['latency']:>10.3f}s  "
#             f"{result['throughput_tps']:>8.3f}  "
#             f"{result['tokens_generated']:>5}  "
#             f"{peak_r:>6.2f} GB"
#         )

#     # unload model to free RAM before loading the next one
#     del model, tq_engine, tokenizer
#     gc.collect()
#     print(f"\n  ♻️   Model unloaded  |  RAM now: {global_ram.current_gb:.2f} GB")

#     # aggregate
#     ttfts     = [r["ttft"]           for r in run_results]
#     latencies = [r["latency"]        for r in run_results]
#     tpss      = [r["throughput_tps"] for r in run_results]
#     rams      = [r["peak_run_ram_gb"] for r in run_results]

#     layers        = spec["layers"]
#     fp32_kv_mb    = CONTEXT_WINDOW * 2 * layers * 4 / (1024 ** 2)
#     turbo_kv_mb   = CONTEXT_WINDOW * layers * (key_bits + val_bits) / 8 / (1024 ** 2)
#     compress_ratio = fp32_kv_mb / turbo_kv_mb

#     return {
#         "label"          : label,
#         "model_key"      : model_key,
#         "n_layers"       : layers,
#         "n_input_tokens" : n_tokens,
#         "runs"           : run_results,
#         "ttft_mean"      : round(statistics.mean(ttfts), 4),
#         "ttft_min"       : round(min(ttfts), 4),
#         "ttft_max"       : round(max(ttfts), 4),
#         "ttft_std"       : round(statistics.stdev(ttfts), 4) if len(ttfts) > 1 else None,
#         "latency_mean"   : round(statistics.mean(latencies), 4),
#         "latency_min"    : round(min(latencies), 4),
#         "latency_max"    : round(max(latencies), 4),
#         "latency_std"    : round(statistics.stdev(latencies), 4) if len(latencies) > 1 else None,
#         "tps_mean"       : round(statistics.mean(tpss), 3),
#         "tps_min"        : round(min(tpss), 3),
#         "tps_max"        : round(max(tpss), 3),
#         "tps_std"        : round(statistics.stdev(tpss), 3) if len(tpss) > 1 else None,
#         "ram_mean_gb"    : round(statistics.mean(rams), 3),
#         "fp32_kv_mb"     : round(fp32_kv_mb, 2),
#         "turbo_kv_mb"    : round(turbo_kv_mb, 2),
#         "compress_ratio" : round(compress_ratio, 2),
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # COMPARATIVE REPORT
# # ─────────────────────────────────────────────────────────────────────────────

# def _delta(a, b, higher_is_better=False):
#     """Return a ▲/▼ delta string: positive = E4B is better."""
#     diff = b - a
#     if higher_is_better:
#         arrow = "▲" if diff >= 0 else "▼"
#         pct   = abs(diff) / a * 100 if a else 0
#         return f"{arrow} {abs(diff):+.4g}  ({pct:.1f}%)"
#     else:
#         arrow = "▼" if diff <= 0 else "▲"
#         pct   = abs(diff) / a * 100 if a else 0
#         return f"{arrow} {abs(diff):+.4g}  ({pct:.1f}%)"


# def _std_fmt(v):
#     return f"±{v:.4f}" if v is not None else "   ±—  "


# def print_comparison(e2b: dict, e4b: dict, key_bits: int, val_bits: int):
#     print()
#     print(SEP2)
#     print("  COMPARATIVE ANALYSIS  :  Gemma 4 E2B  vs  Gemma 4 E4B")
#     print(f"  TurboQuant KV Cache  ·  key={key_bits}b  val={val_bits}b  "
#           f"·  context={CONTEXT_WINDOW}  ·  CPU / float32")
#     print(SEP2)

#     # ── Model Architecture ────────────────────────────────────────────────────
#     print(f"\n  {'ARCHITECTURE':<30}  {'E2B':>12}  {'E4B':>12}  {'Δ (E4B vs E2B)':>22}")
#     print(f"  {SEP}")
#     arch_rows = [
#         ("Hidden layers",  e2b["n_layers"],     e4b["n_layers"],    False),
#         ("Input tokens",   e2b["n_input_tokens"],e4b["n_input_tokens"], False),
#     ]
#     for label, va, vb, hib in arch_rows:
#         delta = _delta(va, vb, higher_is_better=hib) if va != vb else "—"
#         print(f"  {label:<30}  {str(va):>12}  {str(vb):>12}  {delta:>22}")

#     # ── Latency Metrics ───────────────────────────────────────────────────────
#     print(f"\n  {'LATENCY METRICS':<30}  {'E2B mean':>12}  {'E4B mean':>12}  {'Δ (E4B vs E2B)':>22}")
#     print(f"  {SEP}")
#     lat_rows = [
#         ("TTFT (s)",           e2b["ttft_mean"],    e4b["ttft_mean"],    False, "ttft_std"),
#         ("Total latency (s)",  e2b["latency_mean"], e4b["latency_mean"], False, "latency_std"),
#     ]
#     for label, va, vb, hib, std_key in lat_rows:
#         std_a = _std_fmt(e2b[std_key])
#         std_b = _std_fmt(e4b[std_key])
#         delta = _delta(va, vb, higher_is_better=hib)
#         print(f"  {label:<30}  {va:>9.4f}s {std_a}  {vb:>9.4f}s {std_b}  {delta:>22}")

#     # ── Throughput ────────────────────────────────────────────────────────────
#     print(f"\n  {'THROUGHPUT':<30}  {'E2B mean':>12}  {'E4B mean':>12}  {'Δ (E4B vs E2B)':>22}")
#     print(f"  {SEP}")
#     std_a = _std_fmt(e2b["tps_std"])
#     std_b = _std_fmt(e4b["tps_std"])
#     delta = _delta(e2b["tps_mean"], e4b["tps_mean"], higher_is_better=True)
#     print(f"  {'Throughput (tok/s)':<30}  {e2b['tps_mean']:>8.3f}  {std_a}  "
#           f"{e4b['tps_mean']:>8.3f}  {std_b}  {delta:>22}")

#     # ── RAM ───────────────────────────────────────────────────────────────────
#     print(f"\n  {'RAM (GB)':<30}  {'E2B':>12}  {'E4B':>12}  {'Δ (E4B vs E2B)':>22}")
#     print(f"  {SEP}")
#     delta_ram = _delta(e2b["ram_mean_gb"], e4b["ram_mean_gb"], higher_is_better=False)
#     print(f"  {'Peak per-run RAM':<30}  {e2b['ram_mean_gb']:>10.3f}G  {e4b['ram_mean_gb']:>10.3f}G  {delta_ram:>22}")

#     # ── KV Cache Compression ──────────────────────────────────────────────────
#     print(f"\n  {'KV CACHE COMPRESSION':<30}  {'E2B':>12}  {'E4B':>12}  {'Note':>22}")
#     print(f"  {SEP}")
#     kv_rows = [
#         ("Float32 KV size (MB)",  e2b["fp32_kv_mb"],    e4b["fp32_kv_mb"],    "baseline"),
#         ("TurboQuant KV (MB)",    e2b["turbo_kv_mb"],   e4b["turbo_kv_mb"],   f"key={key_bits}b val={val_bits}b"),
#         ("Compression ratio",     e2b["compress_ratio"],e4b["compress_ratio"],"fp32 ÷ turbo"),
#     ]
#     for label, va, vb, note in kv_rows:
#         print(f"  {label:<30}  {str(va):>12}  {str(vb):>12}  {note:>22}")

#     # ── Run-level detail ──────────────────────────────────────────────────────
#     print(f"\n  {'PER-RUN DETAIL'}")
#     print(f"  {SEP}")
#     max_runs = max(len(e2b["runs"]), len(e4b["runs"]))
#     hdr = (f"  {'Run':>4}  "
#            f"{'E2B TTFT':>9}  {'E2B Lat':>9}  {'E2B TPS':>9}  "
#            f"    "
#            f"{'E4B TTFT':>9}  {'E4B Lat':>9}  {'E4B TPS':>9}")
#     print(hdr)
#     print(f"  {SEP3[:len(hdr)-2]}")
#     for i in range(max_runs):
#         ra = e2b["runs"][i] if i < len(e2b["runs"]) else None
#         rb = e4b["runs"][i] if i < len(e4b["runs"]) else None
#         a_str = (f"{ra['ttft']:>8.3f}s  {ra['latency']:>8.3f}s  {ra['throughput_tps']:>8.3f}"
#                  if ra else "         —           —           —")
#         b_str = (f"{rb['ttft']:>8.3f}s  {rb['latency']:>8.3f}s  {rb['throughput_tps']:>8.3f}"
#                  if rb else "         —           —           —")
#         print(f"  {i+1:>4}  {a_str}      {b_str}")

#     # ── Verdict ───────────────────────────────────────────────────────────────
#     print()
#     print(SEP2)
#     print("  VERDICT")
#     print(SEP)

#     faster_ttft  = "E2B" if e2b["ttft_mean"] < e4b["ttft_mean"] else "E4B"
#     faster_lat   = "E2B" if e2b["latency_mean"] < e4b["latency_mean"] else "E4B"
#     higher_tps   = "E2B" if e2b["tps_mean"] > e4b["tps_mean"] else "E4B"
#     lower_ram    = "E2B" if e2b["ram_mean_gb"] < e4b["ram_mean_gb"] else "E4B"

#     ttft_gap_pct = abs(e2b["ttft_mean"] - e4b["ttft_mean"]) / min(e2b["ttft_mean"], e4b["ttft_mean"]) * 100
#     lat_gap_pct  = abs(e2b["latency_mean"] - e4b["latency_mean"]) / min(e2b["latency_mean"], e4b["latency_mean"]) * 100
#     tps_gap_pct  = abs(e2b["tps_mean"] - e4b["tps_mean"]) / max(e2b["tps_mean"], e4b["tps_mean"]) * 100
#     ram_gap_pct  = abs(e2b["ram_mean_gb"] - e4b["ram_mean_gb"]) / min(e2b["ram_mean_gb"], e4b["ram_mean_gb"]) * 100

#     print(f"  ⏱  Faster TTFT      → {faster_ttft}  ({ttft_gap_pct:.1f}% gap)")
#     print(f"  🕐  Lower latency   → {faster_lat}  ({lat_gap_pct:.1f}% gap)")
#     print(f"  ⚡  Higher TPS      → {higher_tps}  ({tps_gap_pct:.1f}% gap)")
#     print(f"  💾  Lower RAM       → {lower_ram}  ({ram_gap_pct:.1f}% gap)")
#     print()
#     print(f"  KV Compression: E2B saves {e2b['fp32_kv_mb'] - e2b['turbo_kv_mb']:.1f} MB  "
#           f"({e2b['compress_ratio']}× ratio)  |  "
#           f"E4B saves {e4b['fp32_kv_mb'] - e4b['turbo_kv_mb']:.1f} MB  "
#           f"({e4b['compress_ratio']}× ratio)")

#     if e2b["compress_ratio"] == e4b["compress_ratio"]:
#         print("  ✅  Both models achieve the same compression ratio with TurboQuant.")
#     else:
#         better_cr = "E2B" if e2b["compress_ratio"] > e4b["compress_ratio"] else "E4B"
#         print(f"  ✅  {better_cr} achieves a higher compression ratio with TurboQuant.")

#     print(SEP2)
#     print()


# # ─────────────────────────────────────────────────────────────────────────────
# # CSV EXPORT
# # ─────────────────────────────────────────────────────────────────────────────

# def save_csv(e2b: dict, e4b: dict, path: str = "benchmark_compare_results.csv"):
#     with open(path, "w", newline="") as f:
#         w = csv.writer(f)

#         # ── Summary ────────────────────────────────────────────────────────────
#         w.writerow(["## SUMMARY"])
#         w.writerow(["metric", "e2b_mean", "e4b_mean", "delta", "winner"])
#         summary = [
#             ("ttft_s",        e2b["ttft_mean"],    e4b["ttft_mean"],    False),
#             ("latency_s",     e2b["latency_mean"], e4b["latency_mean"], False),
#             ("throughput_tps",e2b["tps_mean"],     e4b["tps_mean"],     True),
#             ("peak_ram_gb",   e2b["ram_mean_gb"],  e4b["ram_mean_gb"],  False),
#             ("fp32_kv_mb",    e2b["fp32_kv_mb"],   e4b["fp32_kv_mb"],   False),
#             ("turbo_kv_mb",   e2b["turbo_kv_mb"],  e4b["turbo_kv_mb"],  False),
#             ("compress_ratio",e2b["compress_ratio"],e4b["compress_ratio"],True),
#         ]
#         for label, va, vb, hib in summary:
#             diff   = vb - va
#             winner = "E4B" if (diff > 0) == hib else "E2B"
#             w.writerow([label, va, vb, round(diff, 6), winner])

#         # ── Per-run detail ─────────────────────────────────────────────────────
#         w.writerow([])
#         w.writerow(["## PER-RUN DETAIL"])
#         w.writerow(["model", "run", "ttft_s", "latency_s", "throughput_tps",
#                     "tokens_generated", "peak_run_ram_gb"])
#         for r in e2b["runs"]:
#             w.writerow(["E2B", e2b["runs"].index(r)+1,
#                         r["ttft"], r["latency"], r["throughput_tps"],
#                         r["tokens_generated"], r["peak_run_ram_gb"]])
#         for r in e4b["runs"]:
#             w.writerow(["E4B", e4b["runs"].index(r)+1,
#                         r["ttft"], r["latency"], r["throughput_tps"],
#                         r["tokens_generated"], r["peak_run_ram_gb"]])

#     print(f"  💾  Comparison CSV saved → {path}\n")


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(
#         description="Side-by-side benchmark: Gemma 4 E2B vs E4B with TurboQuant KV Cache"
#     )
#     parser.add_argument("--e2b-model", default=MODEL_SPECS["E2B"]["default_path"],
#                         help="Path to Gemma 4 E2B model directory")
#     parser.add_argument("--e4b-model", default=MODEL_SPECS["E4B"]["default_path"],
#                         help="Path to Gemma 4 E4B model directory")
#     parser.add_argument("--runs",      "-r", type=int, default=DEFAULT_RUNS)
#     parser.add_argument("--prompt",    "-p", type=str, default=DEFAULT_PROMPT)
#     parser.add_argument("--key-bits",        type=int, default=DEFAULT_KEY_BITS)
#     parser.add_argument("--val-bits",        type=int, default=DEFAULT_VAL_BITS)
#     parser.add_argument("--skip-e2b",  action="store_true", help="Skip E2B, run E4B only")
#     parser.add_argument("--skip-e4b",  action="store_true", help="Skip E4B, run E2B only")
#     parser.add_argument("--csv",       default="benchmark_compare_results.csv")
#     args = parser.parse_args()

#     if args.skip_e2b and args.skip_e4b:
#         sys.exit("❌  Cannot skip both models.")

#     models_to_run = []
#     if not args.skip_e2b:
#         if not os.path.isdir(args.e2b_model):
#             sys.exit(f"❌  E2B model directory not found: {args.e2b_model}")
#         models_to_run.append(("E2B", args.e2b_model))
#     if not args.skip_e4b:
#         if not os.path.isdir(args.e4b_model):
#             sys.exit(f"❌  E4B model directory not found: {args.e4b_model}")
#         models_to_run.append(("E4B", args.e4b_model))

#     print(SEP2)
#     print("  Gemma 4  E2B vs E4B  ·  TurboQuant KV Cache  ·  CPU-only Benchmark")
#     print(SEP2)
#     print(f"  Runs           : {args.runs}")
#     print(f"  Key bits       : {args.key_bits}  ({2**args.key_bits} centroids)")
#     print(f"  Value bits     : {args.val_bits}  ({2**args.val_bits} centroids)")
#     print(f"  Context window : {CONTEXT_WINDOW} tokens")
#     print(f"  Max new tokens : {MAX_NEW_TOKENS}")
#     print(f"  Device         : CPU  (float32)")
#     print(f"  Prompt         : \"{args.prompt[:70]}{'…' if len(args.prompt) > 70 else ''}\"")
#     print(SEP2)

#     global_ram   = RAMSampler().start()
#     all_results  = {}

#     for model_key, model_path in models_to_run:
#         result = benchmark_model(
#             model_key  = model_key,
#             model_path = model_path,
#             runs       = args.runs,
#             key_bits   = args.key_bits,
#             val_bits   = args.val_bits,
#             prompt     = args.prompt,
#             global_ram = global_ram,
#         )
#         all_results[model_key] = result

#     global_ram.stop()

#     if len(all_results) == 2:
#         print_comparison(all_results["E2B"], all_results["E4B"], args.key_bits, args.val_bits)
#         save_csv(all_results["E2B"], all_results["E4B"], path=args.csv)
#     else:
#         # Single model — just print its summary
#         key = list(all_results.keys())[0]
#         r   = all_results[key]
#         print(f"\n  Single-model run complete: {r['label']}")
#         print(f"  TTFT mean    : {r['ttft_mean']:.4f}s")
#         print(f"  Latency mean : {r['latency_mean']:.4f}s")
#         print(f"  TPS mean     : {r['tps_mean']:.3f}")
#         print(f"  RAM mean     : {r['ram_mean_gb']:.3f} GB")
#         print(f"  Compression  : {r['compress_ratio']}×  "
#               f"({r['fp32_kv_mb']} MB → {r['turbo_kv_mb']} MB)")


# if __name__ == "__main__":
#     main()