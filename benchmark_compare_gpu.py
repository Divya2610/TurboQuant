#!/usr/bin/env python3
"""
benchmark_compare.py — Gemma 4 E2B vs E4B · TurboQuant KV Cache · GPU
=======================================================================
Runs both models sequentially on the same prompt and produces a
side-by-side comparative analysis.

Metrics  :  TTFT  |  Latency  |  Throughput  |  RAM (GB)  |  VRAM (GB)
           KV Cache size  |  Compression ratio  |  Δ% vs baseline (E2B)
Context  :  2 048 tokens (fixed)
Backend  :  CUDA GPU, float16

Fixes applied vs original:
  1.  VRAMSampler.start() calls torch.cuda.set_device() + torch.cuda.init()
      before reset_peak_memory_stats() — fixes "Invalid device argument"
      RuntimeError when the sampler is started before the CUDA context exists.
  2.  main() explicitly initialises CUDA before the first VRAMSampler is
      constructed, as belt-and-suspenders against the same error.
  3.  MODEL_SPECS hardcoded layer counts removed — num_layers is now read
      from model config at runtime via TurboQuantEngine, so it is always
      accurate for any model variant.
  4.  TurboQuantEngine stores self.num_layers, self.num_kv_heads, and
      self.head_dim as instance attributes and exposes estimate_sizes_mb()
      — the original stored num_layers only as a local variable, causing
      AttributeError on external access, and the compression-size formula
      omitted num_kv_heads × head_dim, producing wildly wrong ratios.
  5.  Rotation matrices and Lloyd-Max codebooks built/stored in float16
      (matching the model dtype) — prevents silent float32 upcasts during
      every matmul in the decode loop, saving VRAM and improving speed.
  6.  Lloyd-Max codebook fitting runs on GPU from the start — no CPU→GPU
      copy overhead.
  7.  Warmup run added per model and excluded from results — eliminates
      JIT/cuDNN overhead from the first timed measurement.
  8.  Decode-mask length verified each step — defensive guard against
      silent misalignment across transformers versions.
  9.  benchmark_model captures compression sizes before deleting the engine
      and uses the accurate estimate_sizes_mb() formula.
  10. --skip-e2b / --skip-e4b path-existence checks now only run for the
      model(s) that will actually be used — previously both paths were
      validated unconditionally even when one was skipped.
  11. tokenizer double-load eliminated — the original loaded each tokenizer
      twice (once to build the chat text, once inside benchmark_model).
      Now the chat text is built inside benchmark_model with the same
      tokenizer that is already loaded for inference.
"""

import argparse
import os
import sys
import time
import threading
import statistics
import gc

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

DEFAULT_PROMPT = (
    "Explain the key differences between supervised, unsupervised, and "
    "reinforcement learning. Give one real-world example for each."
)

GEMMA4_CHAT_TEMPLATE = (
    "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
)

# FIX 3: layer counts removed — read from model config at runtime
MODEL_SPECS = {
    "E2B": {"label": "Gemma 4 E2B", "default_path": "./google-gemma-4-E2B"},
    "E4B": {"label": "Gemma 4 E4B", "default_path": "./gemma-4-E4B"},
}

SEP  = "─" * 78
SEP2 = "═" * 78
SEP3 = "┄" * 78


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMICCACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cache_has_old_api(cache: DynamicCache) -> bool:
    return hasattr(cache, "key_cache")


def extract_kv_from_cache(cache: DynamicCache) -> list:
    if _cache_has_old_api(cache):
        result = []
        for k, v in zip(cache.key_cache, cache.value_cache):
            result.append((k, v) if k is not None else None)
        return result
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
            k, v = kv
            new_cache.update(k, v, i)
    return new_cache


# ─────────────────────────────────────────────────────────────────────────────
# RAM / VRAM SAMPLERS
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


class VRAMSampler:
    def __init__(self, device: torch.device):
        self._device = device
        self._peak   = 0.0
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            allocated = torch.cuda.memory_allocated(self._device) / (1024 ** 3)
            if allocated > self._peak:
                self._peak = allocated
            time.sleep(0.1)

    def start(self):
        # FIX 1: ensure CUDA context is initialised before calling
        # reset_peak_memory_stats() — without this, starting VRAMSampler
        # before any tensor/model is placed on the GPU raises:
        #   RuntimeError: Invalid device argument
        torch.cuda.set_device(self._device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(self._device)
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        torch_peak = torch.cuda.max_memory_allocated(self._device) / (1024 ** 3)
        return max(self._peak, torch_peak)

    @property
    def current_gb(self) -> float:
        return torch.cuda.memory_allocated(self._device) / (1024 ** 3)


# ─────────────────────────────────────────────────────────────────────────────
# TURBOQUANT CORE
# ─────────────────────────────────────────────────────────────────────────────

def build_rotation_matrix(
    dim:    int,
    seed:   int,
    dtype:  torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    FIX 5: dtype and device are explicit — QR runs in float32 for stability
    then the result is cast to the target dtype (float16).  Keeps every
    matmul in the decode loop in float16 and avoids silent upcasts.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    G = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q.to(dtype=dtype, device=device)


def fit_lloyd_max(
    n_bits:    int,
    device:    torch.device,          # FIX 6: runs on GPU, no CPU→GPU copy
    n_samples: int = 100_000,
    n_iter:    int = 150,
) -> tuple:
    n_levels  = 2 ** n_bits
    gen       = torch.Generator(device=device)
    gen.manual_seed(0)
    sample    = torch.randn(n_samples, generator=gen, device=device)
    centroids = torch.linspace(-3.0, 3.0, n_levels, device=device)

    for _ in range(n_iter):
        boundaries  = (centroids[:-1] + centroids[1:]) / 2.0
        full_bounds = torch.cat([
            torch.tensor([-1e9], device=device),
            boundaries,
            torch.tensor([1e9],  device=device),
        ])
        new_c = torch.empty_like(centroids)
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
    def __init__(
        self,
        head_dim:  int,
        key_bits:  int,
        val_bits:  int,
        layer_idx: int,
        device:    torch.device,
        dtype:     torch.dtype,   # FIX 5
    ):
        self.head_dim = head_dim
        self.device   = device
        self.dtype    = dtype
        # FIX 5: rotation matrices in the correct dtype from the start
        self.Q_k = build_rotation_matrix(head_dim, seed=layer_idx * 2,     dtype=dtype, device=device)
        self.Q_v = build_rotation_matrix(head_dim, seed=layer_idx * 2 + 1, dtype=dtype, device=device)
        self.k_bounds = None
        self.k_cents  = None
        self.v_bounds = None
        self.v_cents  = None

    def _compress(self, x: torch.Tensor, Q: torch.Tensor, bounds: torch.Tensor):
        y       = x @ Q.T
        scale   = y.std(dim=-1, keepdim=True).clamp(min=1e-8)
        indices = torch.bucketize((y / scale).contiguous(), bounds)
        return indices.to(torch.uint8), scale

    def _decompress(
        self,
        indices: torch.Tensor,
        scale:   torch.Tensor,
        Q:       torch.Tensor,
        cents:   torch.Tensor,
    ) -> torch.Tensor:
        # FIX 5: cast centroids to working dtype before matmul
        return (cents[indices.long()].to(self.dtype) * scale) @ Q

    def compress(self, key: torch.Tensor, value: torch.Tensor) -> dict:
        k_idx, k_scale = self._compress(key,   self.Q_k, self.k_bounds)
        v_idx, v_scale = self._compress(value, self.Q_v, self.v_bounds)
        return {"k_idx": k_idx, "k_scale": k_scale,
                "v_idx": v_idx, "v_scale": v_scale}

    def decompress(self, data: dict) -> tuple:
        key   = self._decompress(data["k_idx"], data["k_scale"], self.Q_k, self.k_cents)
        value = self._decompress(data["v_idx"], data["v_scale"], self.Q_v, self.v_cents)
        return key, value


# ─────────────────────────────────────────────────────────────────────────────
# TURBOQUANT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TurboQuantEngine:
    def __init__(
        self,
        model,
        key_bits: int,
        val_bits: int,
        device:   torch.device,
        dtype:    torch.dtype = torch.float16,  # FIX 5
    ):
        cfg      = model.config
        text_cfg = getattr(cfg, "text_config", cfg)

        # FIX 3 + 4: read from config; store as instance attributes
        self.num_layers = text_cfg.num_hidden_layers
        self.key_bits   = key_bits
        self.val_bits   = val_bits
        self.device     = device
        self.dtype      = dtype

        head_dim = getattr(text_cfg, "head_dim", None)
        if head_dim is None:
            head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
        self.head_dim = head_dim

        # FIX 4: store num_kv_heads for accurate compression estimates
        self.num_kv_heads = getattr(
            text_cfg, "num_key_value_heads",
            text_cfg.num_attention_heads,
        )

        print(f"    Fitting Lloyd-Max codebooks on {device} …", end=" ", flush=True)
        t0 = time.perf_counter()
        # FIX 6: codebooks built on GPU — arrive ready, no .to() needed
        k_bounds, k_cents = fit_lloyd_max(key_bits, device)
        v_bounds, v_cents = fit_lloyd_max(val_bits, device)
        print(f"done in {time.perf_counter() - t0:.2f}s")

        self.compressors = []
        for i in range(self.num_layers):
            lc          = LayerCompressor(
                head_dim, key_bits, val_bits,
                layer_idx=i, device=device, dtype=dtype,  # FIX 5
            )
            lc.k_bounds = k_bounds
            lc.k_cents  = k_cents
            lc.v_bounds = v_bounds
            lc.v_cents  = v_cents
            self.compressors.append(lc)

        print(f"    {self.num_layers} layer compressors ready  "
              f"(head_dim={head_dim}, kv_heads={self.num_kv_heads}, dtype={dtype})")

    def compress_cache(self, cache: DynamicCache) -> list:
        layer_kvs  = extract_kv_from_cache(cache)
        compressed = []
        for i, cmp in enumerate(self.compressors):
            if i < len(layer_kvs) and layer_kvs[i] is not None:
                k, v = layer_kvs[i]
                compressed.append(cmp.compress(k, v))
            else:
                compressed.append(None)
        return compressed

    def decompress_to_cache(self, compressed: list) -> DynamicCache:
        layer_kvs = []
        for cmp, data in zip(self.compressors, compressed):
            layer_kvs.append(cmp.decompress(data) if data is not None else None)
        return build_cache_from_kv(layer_kvs)

    # FIX 4: accurate size estimate — includes num_kv_heads × head_dim and
    # scale-factor overhead; matches the GPU benchmark script formula exactly
    def estimate_sizes_mb(self, context_len: int) -> dict:
        h   = self.num_kv_heads
        d   = self.head_dim
        L   = self.num_layers
        seq = context_len

        float16_bytes = seq * h * d * 2 * 2 * L           # K+V, 2 bytes each
        k_idx_bytes   = seq * h * d * (self.key_bits / 8) * L
        v_idx_bytes   = seq * h * d * (self.val_bits / 8) * L
        scale_bytes   = seq * h * 1 * 2 * 2 * L           # K+V scale, float16
        turbo_bytes   = k_idx_bytes + v_idx_bytes + scale_bytes

        return {
            "float16_mb": float16_bytes / (1024 ** 2),
            "turbo_mb"  : turbo_bytes   / (1024 ** 2),
            "ratio"     : float16_bytes / turbo_bytes if turbo_bytes > 0 else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE BENCHMARK RUN
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_single(model, tokenizer, input_ids, attention_mask, tq_engine, device) -> dict:
    eos_id           = tokenizer.eos_token_id
    tokens_generated = 0
    seq_len          = input_ids.shape[-1]

    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    out  = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    torch.cuda.synchronize(device)
    ttft = time.perf_counter() - t_start

    next_token       = out.logits[:, -1:, :].argmax(dim=-1)
    tokens_generated += 1

    compressed_store = tq_engine.compress_cache(out.past_key_values)
    del out
    gc.collect()
    torch.cuda.empty_cache()

    decode_mask = torch.ones(
        (1, seq_len + 1), dtype=attention_mask.dtype, device=device
    )

    # FIX 8: verify mask length each decode step
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
                     torch.ones((1, pad), dtype=decode_mask.dtype, device=device)],
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
        next_token       = out.logits[:, -1:, :].argmax(dim=-1)
        tokens_generated += 1

        compressed_store = tq_engine.compress_cache(out.past_key_values)
        del out, past_kv
        gc.collect()
        torch.cuda.empty_cache()

        decode_mask = torch.cat(
            [decode_mask,
             torch.ones((1, 1), dtype=decode_mask.dtype, device=device)],
            dim=-1,
        )

    torch.cuda.synchronize(device)
    latency = time.perf_counter() - t_start
    return {
        "ttft"             : round(ttft, 4),
        "latency"          : round(latency, 4),
        "throughput_tps"   : round(tokens_generated / latency if latency > 0 else 0.0, 3),
        "tokens_generated" : tokens_generated,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUN ONE FULL MODEL BENCHMARK  →  returns summary dict
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_model(model_key: str, model_path: str, args, device: torch.device) -> dict:
    """
    Load model, build prompt text, run warmup + timed runs, unload, return dict.

    FIX 11: tokenizer is loaded once here and reused for both chat-template
    building and inference — the original loaded it twice per model.
    """
    spec  = MODEL_SPECS[model_key]
    label = spec["label"]

    print(f"\n  {'─'*10}  {label}  {'─'*10}")
    print(f"  Path : {model_path}")

    ram_load  = RAMSampler().start()
    vram_load = VRAMSampler(device).start()

    # FIX 11: single tokenizer load
    print("  Loading tokenizer …", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    print("done")

    print("  Loading model …", end=" ", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype       = torch.float16,
        device_map        = "auto",
        trust_remote_code = True,
        local_files_only  = True,
    )
    model.eval()
    ram_after_load  = ram_load.current_gb
    vram_after_load = vram_load.current_gb
    print(f"done  |  RAM {ram_after_load:.2f} GB  |  VRAM {vram_after_load:.2f} GB")

    # FIX 5 + 6: pass dtype so engine uses float16 throughout; codebooks on GPU
    tq_engine = TurboQuantEngine(model, args.key_bits, args.val_bits, device, dtype=torch.float16)

    # FIX 3 + 4: layer/head counts read from engine (which got them from config)
    n_layers   = tq_engine.num_layers
    n_kv_heads = tq_engine.num_kv_heads
    print(f"  Layers   : {n_layers}  (from config)")
    print(f"  KV heads : {n_kv_heads}")

    # FIX 9: capture sizes before engine is deleted
    sizes = tq_engine.estimate_sizes_mb(CONTEXT_WINDOW)

    # FIX 11: build chat text with the already-loaded tokenizer
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except (ValueError, AttributeError):
        print("  ℹ️  No chat template — using Gemma 4 turn format.")
        text = GEMMA4_CHAT_TEMPLATE.format(prompt=args.prompt)

    inputs         = tokenizer(text, return_tensors="pt")
    input_ids      = inputs["input_ids"][:, :CONTEXT_WINDOW]
    attention_mask = inputs.get("attention_mask",
                                torch.ones_like(input_ids))[:, :CONTEXT_WINDOW]
    print(f"  Input tokens : {input_ids.shape[-1]}")

    # FIX 7: warmup run — discarded from results
    print("  Warmup (discarded) …", end=" ", flush=True)
    _ = run_single(model, tokenizer, input_ids, attention_mask, tq_engine, device)
    print("done")

    print(f"  Running {args.runs} benchmark run(s) …")

    run_results = []
    peak_vrams  = []
    peak_rams   = []

    for i in range(args.runs):
        run_ram  = RAMSampler().start()
        run_vram = VRAMSampler(device).start()
        r        = run_single(model, tokenizer, input_ids, attention_mask, tq_engine, device)
        pr       = run_ram.stop()
        pv       = run_vram.stop()
        run_results.append(r)
        peak_rams.append(pr)
        peak_vrams.append(pv)
        print(f"    Run {i+1}  TTFT {r['ttft']:.3f}s  Latency {r['latency']:.3f}s  "
              f"TPS {r['throughput_tps']:.2f}  Tok {r['tokens_generated']}  "
              f"RAM {pr:.2f}GB  VRAM {pv:.2f}GB")

    peak_total_ram  = ram_load.stop()
    peak_total_vram = vram_load.stop()

    # Free model memory before loading next
    del model, tq_engine, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    ttfts     = [r["ttft"]             for r in run_results]
    latencies = [r["latency"]          for r in run_results]
    tpss      = [r["throughput_tps"]   for r in run_results]
    tokens    = [r["tokens_generated"] for r in run_results]

    return {
        "model_key"        : model_key,
        "display_name"     : label,
        "num_layers"       : n_layers,
        "num_kv_heads"     : n_kv_heads,
        "ttft_mean"        : statistics.mean(ttfts),
        "ttft_min"         : min(ttfts),
        "ttft_max"         : max(ttfts),
        "ttft_sd"          : statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
        "lat_mean"         : statistics.mean(latencies),
        "lat_min"          : min(latencies),
        "lat_max"          : max(latencies),
        "lat_sd"           : statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "tps_mean"         : statistics.mean(tpss),
        "tps_min"          : min(tpss),
        "tps_max"          : max(tpss),
        "tps_sd"           : statistics.stdev(tpss) if len(tpss) > 1 else 0.0,
        "tokens_mean"      : statistics.mean(tokens),
        "peak_ram_gb"      : peak_total_ram,
        "peak_vram_gb"     : peak_total_vram,
        "load_ram_gb"      : ram_after_load,
        "load_vram_gb"     : vram_after_load,
        # FIX 9: accurate size estimates from estimate_sizes_mb()
        "float16_kv_mb"    : sizes["float16_mb"],
        "turbo_kv_mb"      : sizes["turbo_mb"],
        "compression_ratio": sizes["ratio"],
        "run_results"      : run_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPARATIVE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def delta(a, b, lower_is_better=True):
    """Return formatted Δ% string."""
    if a == 0:
        return "     n/a"
    pct = (b - a) / abs(a) * 100
    if lower_is_better:
        arrow = "▼" if pct < 0 else "▲"
        sign  = "+" if pct > 0 else ""
    else:
        arrow = "▲" if pct > 0 else "▼"
        sign  = "+" if pct > 0 else ""
    return f"{arrow} {sign}{pct:.1f}%"


def print_comparison(e2b: dict, e4b: dict, args):
    W = 20   # column width

    def row(label, v_e2b, v_e4b, fmt, lib=True):
        d = delta(v_e2b, v_e4b, lower_is_better=lib)
        print(f"  {label:<28}  {fmt.format(v_e2b):>{W}}  {fmt.format(v_e4b):>{W}}  {d:>12}")

    print()
    print(SEP2)
    print("  COMPARATIVE ANALYSIS  —  Gemma 4 E2B  vs  Gemma 4 E4B")
    print(f"  TurboQuant KV Cache · GPU (float16) · {CONTEXT_WINDOW}-token context · {args.runs} run(s)")
    print(SEP2)
    print(f"  {'METRIC':<28}  {'E2B (baseline)':>{W}}  {'E4B':>{W}}  {'Δ E4B vs E2B':>12}")
    print(SEP)

    # ── Architecture ──────────────────────────────────────────────────────────
    print(f"\n  ── Architecture")
    print(f"  {'Layers':<28}  {e2b['num_layers']:>{W}}  {e4b['num_layers']:>{W}}  {'':>12}")
    print(f"  {'KV heads':<28}  {e2b['num_kv_heads']:>{W}}  {e4b['num_kv_heads']:>{W}}  {'':>12}")

    # ── Latency / Speed ───────────────────────────────────────────────────────
    print(f"\n  ── Latency / Speed")
    row("TTFT mean (s)",          e2b["ttft_mean"],  e4b["ttft_mean"],  "{:.4f}s", lib=True)
    row("TTFT min  (s)",          e2b["ttft_min"],   e4b["ttft_min"],   "{:.4f}s", lib=True)
    row("TTFT max  (s)",          e2b["ttft_max"],   e4b["ttft_max"],   "{:.4f}s", lib=True)
    row("Latency mean (s)",       e2b["lat_mean"],   e4b["lat_mean"],   "{:.4f}s", lib=True)
    row("Latency min  (s)",       e2b["lat_min"],    e4b["lat_min"],    "{:.4f}s", lib=True)
    row("Latency max  (s)",       e2b["lat_max"],    e4b["lat_max"],    "{:.4f}s", lib=True)

    # ── Throughput ────────────────────────────────────────────────────────────
    print(f"\n  ── Throughput")
    row("TPS mean (tok/s)",        e2b["tps_mean"],    e4b["tps_mean"],    "{:.3f}",  lib=False)
    row("TPS min  (tok/s)",        e2b["tps_min"],     e4b["tps_min"],     "{:.3f}",  lib=False)
    row("TPS max  (tok/s)",        e2b["tps_max"],     e4b["tps_max"],     "{:.3f}",  lib=False)
    row("Tokens generated (mean)", e2b["tokens_mean"], e4b["tokens_mean"], "{:.1f}",  lib=False)

    # ── Memory ────────────────────────────────────────────────────────────────
    print(f"\n  ── Memory")
    row("Model load RAM  (GB)",   e2b["load_ram_gb"],  e4b["load_ram_gb"],  "{:.3f}GB", lib=True)
    row("Model load VRAM (GB)",   e2b["load_vram_gb"], e4b["load_vram_gb"], "{:.3f}GB", lib=True)
    row("Peak RAM  (GB)",         e2b["peak_ram_gb"],  e4b["peak_ram_gb"],  "{:.3f}GB", lib=True)
    row("Peak VRAM (GB)",         e2b["peak_vram_gb"], e4b["peak_vram_gb"], "{:.3f}GB", lib=True)

    # ── KV Cache Compression ──────────────────────────────────────────────────
    print(f"\n  ── TurboQuant KV Cache  (key={args.key_bits}bit / val={args.val_bits}bit)")
    row("Float16 KV size  (MB)",  e2b["float16_kv_mb"],    e4b["float16_kv_mb"],    "{:.1f}MB",  lib=True)
    row("TurboQuant size  (MB)",  e2b["turbo_kv_mb"],      e4b["turbo_kv_mb"],      "{:.2f}MB",  lib=True)
    row("Compression ratio",      e2b["compression_ratio"],e4b["compression_ratio"],"{:.1f}×",   lib=False)

    print()
    print(SEP2)

    # ── Winner Summary ────────────────────────────────────────────────────────
    print("  SUMMARY")
    print(SEP)
    faster_ttft  = "E2B" if e2b["ttft_mean"]    < e4b["ttft_mean"]    else "E4B"
    faster_lat   = "E2B" if e2b["lat_mean"]     < e4b["lat_mean"]     else "E4B"
    higher_tps   = "E2B" if e2b["tps_mean"]     > e4b["tps_mean"]     else "E4B"
    lower_vram   = "E2B" if e2b["peak_vram_gb"] < e4b["peak_vram_gb"] else "E4B"

    ttft_diff_pct = abs(e2b["ttft_mean"]    - e4b["ttft_mean"])    / max(e2b["ttft_mean"],    1e-9) * 100
    lat_diff_pct  = abs(e2b["lat_mean"]     - e4b["lat_mean"])     / max(e2b["lat_mean"],     1e-9) * 100
    tps_diff_pct  = abs(e2b["tps_mean"]     - e4b["tps_mean"])     / max(e2b["tps_mean"],     1e-9) * 100
    vram_diff_pct = abs(e2b["peak_vram_gb"] - e4b["peak_vram_gb"]) / max(e2b["peak_vram_gb"], 1e-9) * 100

    print(f"  Faster TTFT       : {faster_ttft}  ({ttft_diff_pct:.1f}% difference)")
    print(f"  Lower latency     : {faster_lat}  ({lat_diff_pct:.1f}% difference)")
    print(f"  Higher throughput : {higher_tps}  ({tps_diff_pct:.1f}% difference)")
    print(f"  Lower VRAM usage  : {lower_vram}  ({vram_diff_pct:.1f}% difference)")
    print(SEP2)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(e2b: dict, e4b: dict, args):
    csv_path = "benchmark_compare_results.csv"
    with open(csv_path, "w") as f:
        f.write("metric,E2B,E4B,delta_pct\n")

        def csv_row(label, v1, v2):
            d = (v2 - v1) / abs(v1) * 100 if v1 != 0 else 0.0
            f.write(f"{label},{v1:.6f},{v2:.6f},{d:+.2f}\n")

        csv_row("ttft_mean_s",        e2b["ttft_mean"],         e4b["ttft_mean"])
        csv_row("ttft_min_s",         e2b["ttft_min"],          e4b["ttft_min"])
        csv_row("ttft_max_s",         e2b["ttft_max"],          e4b["ttft_max"])
        csv_row("latency_mean_s",     e2b["lat_mean"],          e4b["lat_mean"])
        csv_row("latency_min_s",      e2b["lat_min"],           e4b["lat_min"])
        csv_row("latency_max_s",      e2b["lat_max"],           e4b["lat_max"])
        csv_row("tps_mean",           e2b["tps_mean"],          e4b["tps_mean"])
        csv_row("tps_min",            e2b["tps_min"],           e4b["tps_min"])
        csv_row("tps_max",            e2b["tps_max"],           e4b["tps_max"])
        csv_row("tokens_mean",        e2b["tokens_mean"],       e4b["tokens_mean"])
        csv_row("load_ram_gb",        e2b["load_ram_gb"],       e4b["load_ram_gb"])
        csv_row("load_vram_gb",       e2b["load_vram_gb"],      e4b["load_vram_gb"])
        csv_row("peak_ram_gb",        e2b["peak_ram_gb"],       e4b["peak_ram_gb"])
        csv_row("peak_vram_gb",       e2b["peak_vram_gb"],      e4b["peak_vram_gb"])
        csv_row("float16_kv_mb",      e2b["float16_kv_mb"],     e4b["float16_kv_mb"])
        csv_row("turbo_kv_mb",        e2b["turbo_kv_mb"],       e4b["turbo_kv_mb"])
        csv_row("compression_ratio",  e2b["compression_ratio"], e4b["compression_ratio"])

        # Per-run raw data
        f.write("\n--- E2B raw runs ---\n")
        f.write("run,ttft_s,latency_s,tps,tokens\n")
        for i, r in enumerate(e2b["run_results"], 1):
            f.write(f"{i},{r['ttft']},{r['latency']},{r['throughput_tps']},{r['tokens_generated']}\n")

        f.write("\n--- E4B raw runs ---\n")
        f.write("run,ttft_s,latency_s,tps,tokens\n")
        for i, r in enumerate(e4b["run_results"], 1):
            f.write(f"{i},{r['ttft']},{r['latency']},{r['throughput_tps']},{r['tokens_generated']}\n")

    print(f"  💾  Results saved → {csv_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comparative GPU benchmark: Gemma 4 E2B vs E4B"
    )
    parser.add_argument("--model-e2b",  default=MODEL_SPECS["E2B"]["default_path"],
                        help="Path to E2B model directory")
    parser.add_argument("--model-e4b",  default=MODEL_SPECS["E4B"]["default_path"],
                        help="Path to E4B model directory")
    parser.add_argument("--runs",  "-r", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--prompt","-p", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--key-bits",   type=int,  default=DEFAULT_KEY_BITS)
    parser.add_argument("--val-bits",   type=int,  default=DEFAULT_VAL_BITS)
    parser.add_argument("--skip-e2b",   action="store_true",
                        help="Skip E2B and only run E4B")
    parser.add_argument("--skip-e4b",   action="store_true",
                        help="Skip E4B and only run E2B")
    args = parser.parse_args()

    if args.skip_e2b and args.skip_e4b:
        sys.exit("❌  Cannot skip both models.")

    if not torch.cuda.is_available():
        sys.exit("❌  No CUDA GPU detected. Please run on a machine with a GPU.")

    device = torch.device("cuda:0")

    # FIX 2: initialise CUDA context before any VRAMSampler is constructed
    torch.cuda.set_device(device)
    torch.cuda.init()

    # FIX 10: only validate paths for models that will actually be run
    if not args.skip_e2b and not os.path.isdir(args.model_e2b):
        sys.exit(f"❌  E2B model directory not found: {args.model_e2b}")
    if not args.skip_e4b and not os.path.isdir(args.model_e4b):
        sys.exit(f"❌  E4B model directory not found: {args.model_e4b}")

    print(SEP2)
    print("  Gemma 4 E2B vs E4B  ·  TurboQuant KV Cache  ·  Comparative GPU Benchmark")
    print(SEP2)
    print(f"  GPU            : {torch.cuda.get_device_name(device)}")
    print(f"  Key bits       : {args.key_bits}  ({2**args.key_bits} centroids)")
    print(f"  Value bits     : {args.val_bits}  ({2**args.val_bits} centroids)")
    print(f"  Context window : {CONTEXT_WINDOW} tokens")
    print(f"  Max new tokens : {MAX_NEW_TOKENS}")
    print(f"  Runs per model : {args.runs}  (+1 warmup per model, discarded)")
    print(f"  Prompt         : \"{args.prompt[:72]}{'…' if len(args.prompt) > 72 else ''}\"")
    print(SEP2)

    e2b_results = None
    e4b_results = None

    if not args.skip_e2b:
        e2b_results = benchmark_model("E2B", args.model_e2b, args, device)

    if not args.skip_e4b:
        e4b_results = benchmark_model("E4B", args.model_e4b, args, device)

    if e2b_results and e4b_results:
        print_comparison(e2b_results, e4b_results, args)
        save_csv(e2b_results, e4b_results, args)
    elif e2b_results:
        r = e2b_results
        print(f"\n  Single-model run complete: {r['display_name']}")
        print(f"  Layers       : {r['num_layers']}")
        print(f"  KV heads     : {r['num_kv_heads']}")
        print(f"  TTFT mean    : {r['ttft_mean']:.4f}s")
        print(f"  Latency mean : {r['lat_mean']:.4f}s")
        print(f"  TPS mean     : {r['tps_mean']:.3f}")
        print(f"  Peak VRAM    : {r['peak_vram_gb']:.3f} GB")
        print(f"  Compression  : {r['compression_ratio']:.2f}×  "
              f"({r['float16_kv_mb']:.1f} MB → {r['turbo_kv_mb']:.2f} MB)")
    elif e4b_results:
        r = e4b_results
        print(f"\n  Single-model run complete: {r['display_name']}")
        print(f"  Layers       : {r['num_layers']}")
        print(f"  KV heads     : {r['num_kv_heads']}")
        print(f"  TTFT mean    : {r['ttft_mean']:.4f}s")
        print(f"  Latency mean : {r['lat_mean']:.4f}s")
        print(f"  TPS mean     : {r['tps_mean']:.3f}")
        print(f"  Peak VRAM    : {r['peak_vram_gb']:.3f} GB")
        print(f"  Compression  : {r['compression_ratio']:.2f}×  "
              f"({r['float16_kv_mb']:.1f} MB → {r['turbo_kv_mb']:.2f} MB)")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# benchmark_compare.py — Gemma 4 E2B vs E4B · TurboQuant KV Cache · GPU
# =======================================================================
# Runs both models sequentially on the same prompt and produces a
# side-by-side comparative analysis.

# Metrics  :  TTFT  |  Latency  |  Throughput  |  RAM (GB)  |  VRAM (GB)
#            KV Cache size  |  Compression ratio  |  Δ% vs baseline (E2B)
# Context  :  2 048 tokens (fixed)
# Backend  :  CUDA GPU, float16
# """

# import argparse
# import os
# import sys
# import time
# import threading
# import statistics
# import gc

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

# DEFAULT_PROMPT = (
#     "Explain the key differences between supervised, unsupervised, and "
#     "reinforcement learning. Give one real-world example for each."
# )

# GEMMA4_CHAT_TEMPLATE = (
#     "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
# )

# # Model specs: (display_name, default_path, num_layers)
# MODEL_SPECS = {
#     "E2B": ("Gemma 4 E2B", "./google-gemma-4-E2B", 35),
#     "E4B": ("Gemma 4 E4B", "./gemma-4-E4B", 42),
# }

# SEP  = "─" * 78
# SEP2 = "═" * 78
# SEP3 = "┄" * 78


# # ─────────────────────────────────────────────────────────────────────────────
# # DYNAMICCACHE HELPERS
# # ─────────────────────────────────────────────────────────────────────────────

# def _cache_has_old_api(cache: DynamicCache) -> bool:
#     return hasattr(cache, "key_cache")


# def extract_kv_from_cache(cache: DynamicCache) -> list:
#     if _cache_has_old_api(cache):
#         result = []
#         for k, v in zip(cache.key_cache, cache.value_cache):
#             result.append((k, v) if k is not None else None)
#         return result
#     else:
#         result = []
#         for layer in cache.layers:
#             if layer is None:
#                 result.append(None)
#             else:
#                 k = getattr(layer, "key", None)
#                 v = getattr(layer, "value", None)
#                 if k is None:
#                     try:
#                         k, v = layer.key_cache, layer.value_cache
#                     except AttributeError:
#                         result.append(None)
#                         continue
#                 result.append((k, v))
#         return result


# def build_cache_from_kv(layer_kvs: list) -> DynamicCache:
#     new_cache = DynamicCache()
#     for i, kv in enumerate(layer_kvs):
#         if kv is not None:
#             k, v = kv
#             new_cache.update(k, v, i)
#     return new_cache


# # ─────────────────────────────────────────────────────────────────────────────
# # RAM / VRAM SAMPLERS
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


# class VRAMSampler:
#     def __init__(self, device: torch.device):
#         self._device = device
#         self._peak   = 0.0
#         self._stop   = threading.Event()
#         self._thread = threading.Thread(target=self._poll, daemon=True)

#     def _poll(self):
#         while not self._stop.is_set():
#             allocated = torch.cuda.memory_allocated(self._device) / (1024 ** 3)
#             if allocated > self._peak:
#                 self._peak = allocated
#             time.sleep(0.1)

#     def start(self):
#         torch.cuda.reset_peak_memory_stats(self._device)
#         self._thread.start()
#         return self

#     def stop(self) -> float:
#         self._stop.set()
#         self._thread.join()
#         torch_peak = torch.cuda.max_memory_allocated(self._device) / (1024 ** 3)
#         return max(self._peak, torch_peak)

#     @property
#     def current_gb(self) -> float:
#         return torch.cuda.memory_allocated(self._device) / (1024 ** 3)


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
#     def __init__(self, head_dim: int, key_bits: int, val_bits: int, layer_idx: int, device: torch.device):
#         self.head_dim = head_dim
#         self.device   = device
#         self.Q_k      = build_rotation_matrix(head_dim, seed=layer_idx * 2).to(device)
#         self.Q_v      = build_rotation_matrix(head_dim, seed=layer_idx * 2 + 1).to(device)
#         self.k_bounds = None
#         self.k_cents  = None
#         self.v_bounds = None
#         self.v_cents  = None

#     def _compress(self, x, Q, bounds):
#         y       = x @ Q.T
#         scale   = y.std(dim=-1, keepdim=True).clamp(min=1e-8)
#         indices = torch.bucketize((y / scale).contiguous(), bounds)
#         return indices.to(torch.uint8), scale

#     def _decompress(self, indices, scale, Q, cents):
#         return (cents[indices.long()] * scale) @ Q

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
#     def __init__(self, model, key_bits: int, val_bits: int, device: torch.device):
#         cfg      = model.config
#         text_cfg = getattr(cfg, "text_config", cfg)

#         num_layers    = text_cfg.num_hidden_layers
#         self.key_bits = key_bits
#         self.val_bits = val_bits
#         self.device   = device

#         head_dim = getattr(text_cfg, "head_dim", None)
#         if head_dim is None:
#             head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
#         self.head_dim = head_dim

#         print(f"    Fitting Lloyd-Max codebooks …", end=" ", flush=True)
#         t0 = time.perf_counter()
#         k_bounds, k_cents = fit_lloyd_max(key_bits)
#         v_bounds, v_cents = fit_lloyd_max(val_bits)
#         k_bounds = k_bounds.to(device); k_cents = k_cents.to(device)
#         v_bounds = v_bounds.to(device); v_cents = v_cents.to(device)
#         print(f"done in {time.perf_counter() - t0:.2f}s")

#         self.compressors = []
#         for i in range(num_layers):
#             lc          = LayerCompressor(head_dim, key_bits, val_bits, layer_idx=i, device=device)
#             lc.k_bounds = k_bounds; lc.k_cents = k_cents
#             lc.v_bounds = v_bounds; lc.v_cents = v_cents
#             self.compressors.append(lc)

#         print(f"    {num_layers} layer compressors ready  (head_dim={head_dim})")

#     def compress_cache(self, cache: DynamicCache) -> list:
#         layer_kvs  = extract_kv_from_cache(cache)
#         compressed = []
#         for i, cmp in enumerate(self.compressors):
#             if i < len(layer_kvs) and layer_kvs[i] is not None:
#                 k, v = layer_kvs[i]
#                 compressed.append(cmp.compress(k, v))
#             else:
#                 compressed.append(None)
#         return compressed

#     def decompress_to_cache(self, compressed: list) -> DynamicCache:
#         layer_kvs = []
#         for cmp, data in zip(self.compressors, compressed):
#             if data is not None:
#                 layer_kvs.append(cmp.decompress(data))
#             else:
#                 layer_kvs.append(None)
#         return build_cache_from_kv(layer_kvs)


# # ─────────────────────────────────────────────────────────────────────────────
# # SINGLE BENCHMARK RUN
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.inference_mode()
# def run_single(model, tokenizer, input_ids, attention_mask, tq_engine, device) -> dict:
#     eos_id           = tokenizer.eos_token_id
#     tokens_generated = 0

#     input_ids      = input_ids.to(device)
#     attention_mask = attention_mask.to(device)

#     torch.cuda.synchronize(device)
#     t_start = time.perf_counter()

#     out  = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
#     torch.cuda.synchronize(device)
#     ttft = time.perf_counter() - t_start

#     next_token       = out.logits[:, -1:, :].argmax(dim=-1)
#     tokens_generated += 1

#     compressed_store = tq_engine.compress_cache(out.past_key_values)
#     del out
#     gc.collect()
#     torch.cuda.empty_cache()

#     decode_mask = torch.ones((1, input_ids.shape[-1] + 1), dtype=attention_mask.dtype, device=device)

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
#         next_token       = out.logits[:, -1:, :].argmax(dim=-1)
#         tokens_generated += 1

#         compressed_store = tq_engine.compress_cache(out.past_key_values)
#         del out, past_kv
#         gc.collect()
#         torch.cuda.empty_cache()

#         decode_mask = torch.cat(
#             [decode_mask, torch.ones((1, 1), dtype=decode_mask.dtype, device=device)], dim=-1
#         )

#     torch.cuda.synchronize(device)
#     latency = time.perf_counter() - t_start
#     return {
#         "ttft"             : round(ttft, 4),
#         "latency"          : round(latency, 4),
#         "throughput_tps"   : round(tokens_generated / latency if latency > 0 else 0.0, 3),
#         "tokens_generated" : tokens_generated,
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # RUN ONE FULL MODEL BENCHMARK  →  returns summary dict
# # ─────────────────────────────────────────────────────────────────────────────

# def benchmark_model(model_key, model_path, args, device, text) -> dict:
#     display_name, _, num_layers = MODEL_SPECS[model_key]

#     print(f"\n  {'─'*10}  {display_name}  {'─'*10}")
#     print(f"  Path : {model_path}")

#     ram_load  = RAMSampler().start()
#     vram_load = VRAMSampler(device).start()

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path, trust_remote_code=True, local_files_only=True,
#     )
#     print(f"  Loading model …", end=" ", flush=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype       = torch.float16,
#         device_map        = "auto",
#         trust_remote_code = True,
#         local_files_only  = True,
#     )
#     model.eval()
#     ram_after_load  = ram_load.current_gb
#     vram_after_load = vram_load.current_gb
#     print(f"done  |  RAM {ram_after_load:.2f} GB  |  VRAM {vram_after_load:.2f} GB")

#     tq_engine = TurboQuantEngine(model, args.key_bits, args.val_bits, device)

#     inputs         = tokenizer(text, return_tensors="pt")
#     input_ids      = inputs["input_ids"][:, :CONTEXT_WINDOW]
#     attention_mask = inputs.get("attention_mask",
#                                 torch.ones_like(input_ids))[:, :CONTEXT_WINDOW]
#     print(f"  Input tokens : {input_ids.shape[-1]}")
#     print(f"  Running {args.runs} benchmark run(s) …")

#     run_results = []
#     peak_vrams  = []
#     peak_rams   = []

#     for i in range(args.runs):
#         run_ram  = RAMSampler().start()
#         run_vram = VRAMSampler(device).start()
#         r        = run_single(model, tokenizer, input_ids, attention_mask, tq_engine, device)
#         pr       = run_ram.stop()
#         pv       = run_vram.stop()
#         run_results.append(r)
#         peak_rams.append(pr)
#         peak_vrams.append(pv)
#         print(f"    Run {i+1}  TTFT {r['ttft']:.3f}s  Latency {r['latency']:.3f}s  "
#               f"TPS {r['throughput_tps']:.2f}  Tok {r['tokens_generated']}  "
#               f"RAM {pr:.2f}GB  VRAM {pv:.2f}GB")

#     peak_total_ram  = ram_load.stop()
#     peak_total_vram = vram_load.stop()

#     # Free model memory before loading next
#     del model
#     gc.collect()
#     torch.cuda.empty_cache()

#     ttfts     = [r["ttft"]           for r in run_results]
#     latencies = [r["latency"]        for r in run_results]
#     tpss      = [r["throughput_tps"] for r in run_results]
#     tokens    = [r["tokens_generated"] for r in run_results]

#     float32_kv_mb = CONTEXT_WINDOW * 2 * num_layers * 4 / (1024 ** 2)
#     turbo_kv_mb   = CONTEXT_WINDOW * num_layers * (args.key_bits + args.val_bits) / 8 / (1024 ** 2)

#     return {
#         "model_key"        : model_key,
#         "display_name"     : display_name,
#         "num_layers"       : num_layers,
#         "ttft_mean"        : statistics.mean(ttfts),
#         "ttft_min"         : min(ttfts),
#         "ttft_max"         : max(ttfts),
#         "ttft_sd"          : statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
#         "lat_mean"         : statistics.mean(latencies),
#         "lat_min"          : min(latencies),
#         "lat_max"          : max(latencies),
#         "lat_sd"           : statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
#         "tps_mean"         : statistics.mean(tpss),
#         "tps_min"          : min(tpss),
#         "tps_max"          : max(tpss),
#         "tps_sd"           : statistics.stdev(tpss) if len(tpss) > 1 else 0.0,
#         "tokens_mean"      : statistics.mean(tokens),
#         "peak_ram_gb"      : peak_total_ram,
#         "peak_vram_gb"     : peak_total_vram,
#         "load_ram_gb"      : ram_after_load,
#         "load_vram_gb"     : vram_after_load,
#         "float32_kv_mb"    : float32_kv_mb,
#         "turbo_kv_mb"      : turbo_kv_mb,
#         "compression_ratio": float32_kv_mb / turbo_kv_mb,
#         "run_results"      : run_results,
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # COMPARATIVE REPORT
# # ─────────────────────────────────────────────────────────────────────────────

# def delta(a, b, lower_is_better=True):
#     """Return formatted Δ% string: green arrow if improved, red if worse."""
#     if a == 0:
#         return "     n/a"
#     pct = (b - a) / abs(a) * 100
#     if lower_is_better:
#         arrow = "▼" if pct < 0 else "▲"
#         sign  = "+" if pct > 0 else ""
#     else:
#         arrow = "▲" if pct > 0 else "▼"
#         sign  = "+" if pct > 0 else ""
#     return f"{arrow} {sign}{pct:.1f}%"


# def print_comparison(e2b: dict, e4b: dict, args):
#     W = 20   # column width

#     def row(label, v_e2b, v_e4b, fmt, lib=True):
#         d = delta(v_e2b, v_e4b, lower_is_better=lib)
#         print(f"  {label:<28}  {fmt.format(v_e2b):>{W}}  {fmt.format(v_e4b):>{W}}  {d:>12}")

#     print()
#     print(SEP2)
#     print("  COMPARATIVE ANALYSIS  —  Gemma 4 E2B  vs  Gemma 4 E4B")
#     print(f"  TurboQuant KV Cache · GPU (float16) · {CONTEXT_WINDOW}-token context · {args.runs} run(s)")
#     print(SEP2)
#     print(f"  {'METRIC':<28}  {'E2B (baseline)':>{W}}  {'E4B':>{W}}  {'Δ E4B vs E2B':>12}")
#     print(SEP)

#     # ── Architecture ──────────────────────────────────────────────────────────
#     print(f"\n  ── Architecture")
#     print(f"  {'Layers':<28}  {e2b['num_layers']:>{W}}  {e4b['num_layers']:>{W}}  {'':>12}")

#     # ── Latency / Speed ───────────────────────────────────────────────────────
#     print(f"\n  ── Latency / Speed")
#     row("TTFT mean (s)",          e2b["ttft_mean"],  e4b["ttft_mean"],  "{:.4f}s", lib=True)
#     row("TTFT min  (s)",          e2b["ttft_min"],   e4b["ttft_min"],   "{:.4f}s", lib=True)
#     row("TTFT max  (s)",          e2b["ttft_max"],   e4b["ttft_max"],   "{:.4f}s", lib=True)
#     row("Latency mean (s)",       e2b["lat_mean"],   e4b["lat_mean"],   "{:.4f}s", lib=True)
#     row("Latency min  (s)",       e2b["lat_min"],    e4b["lat_min"],    "{:.4f}s", lib=True)
#     row("Latency max  (s)",       e2b["lat_max"],    e4b["lat_max"],    "{:.4f}s", lib=True)

#     # ── Throughput ────────────────────────────────────────────────────────────
#     print(f"\n  ── Throughput")
#     row("TPS mean (tok/s)",       e2b["tps_mean"],   e4b["tps_mean"],   "{:.3f}",  lib=False)
#     row("TPS min  (tok/s)",       e2b["tps_min"],    e4b["tps_min"],    "{:.3f}",  lib=False)
#     row("TPS max  (tok/s)",       e2b["tps_max"],    e4b["tps_max"],    "{:.3f}",  lib=False)
#     row("Tokens generated (mean)",e2b["tokens_mean"],e4b["tokens_mean"],"{:.1f}",  lib=False)

#     # ── Memory ────────────────────────────────────────────────────────────────
#     print(f"\n  ── Memory")
#     row("Model load RAM  (GB)",   e2b["load_ram_gb"],  e4b["load_ram_gb"],  "{:.3f}GB", lib=True)
#     row("Model load VRAM (GB)",   e2b["load_vram_gb"], e4b["load_vram_gb"], "{:.3f}GB", lib=True)
#     row("Peak RAM  (GB)",         e2b["peak_ram_gb"],  e4b["peak_ram_gb"],  "{:.3f}GB", lib=True)
#     row("Peak VRAM (GB)",         e2b["peak_vram_gb"], e4b["peak_vram_gb"], "{:.3f}GB", lib=True)

#     # ── KV Cache Compression ──────────────────────────────────────────────────
#     print(f"\n  ── TurboQuant KV Cache  (key={args.key_bits}bit / val={args.val_bits}bit)")
#     row("Float32 KV size  (MB)",  e2b["float32_kv_mb"],  e4b["float32_kv_mb"],  "{:.1f}MB", lib=True)
#     row("TurboQuant size  (MB)",  e2b["turbo_kv_mb"],    e4b["turbo_kv_mb"],    "{:.2f}MB", lib=True)
#     row("Compression ratio",      e2b["compression_ratio"], e4b["compression_ratio"], "{:.1f}×", lib=False)

#     print()
#     print(SEP2)

#     # ── Winner Summary ────────────────────────────────────────────────────────
#     print("  SUMMARY")
#     print(SEP)
#     faster_ttft  = "E2B" if e2b["ttft_mean"]  < e4b["ttft_mean"]  else "E4B"
#     faster_lat   = "E2B" if e2b["lat_mean"]   < e4b["lat_mean"]   else "E4B"
#     higher_tps   = "E2B" if e2b["tps_mean"]   > e4b["tps_mean"]   else "E4B"
#     lower_vram   = "E2B" if e2b["peak_vram_gb"] < e4b["peak_vram_gb"] else "E4B"

#     ttft_diff_pct  = abs(e2b["ttft_mean"] - e4b["ttft_mean"]) / max(e2b["ttft_mean"], 1e-9) * 100
#     lat_diff_pct   = abs(e2b["lat_mean"]  - e4b["lat_mean"])  / max(e2b["lat_mean"],  1e-9) * 100
#     tps_diff_pct   = abs(e2b["tps_mean"]  - e4b["tps_mean"])  / max(e2b["tps_mean"],  1e-9) * 100
#     vram_diff_pct  = abs(e2b["peak_vram_gb"] - e4b["peak_vram_gb"]) / max(e2b["peak_vram_gb"], 1e-9) * 100

#     print(f"  Faster TTFT       : {faster_ttft}  ({ttft_diff_pct:.1f}% difference)")
#     print(f"  Lower latency     : {faster_lat}  ({lat_diff_pct:.1f}% difference)")
#     print(f"  Higher throughput : {higher_tps}  ({tps_diff_pct:.1f}% difference)")
#     print(f"  Lower VRAM usage  : {lower_vram}  ({vram_diff_pct:.1f}% difference)")
#     print(SEP2)
#     print()


# # ─────────────────────────────────────────────────────────────────────────────
# # CSV EXPORT
# # ─────────────────────────────────────────────────────────────────────────────

# def save_csv(e2b: dict, e4b: dict, args):
#     csv_path = "benchmark_compare_results.csv"
#     with open(csv_path, "w") as f:
#         f.write("metric,E2B,E4B,delta_pct\n")

#         def csv_row(label, v1, v2, lib=True):
#             d = (v2 - v1) / abs(v1) * 100 if v1 != 0 else 0.0
#             f.write(f"{label},{v1:.6f},{v2:.6f},{d:+.2f}\n")

#         csv_row("ttft_mean_s",        e2b["ttft_mean"],    e4b["ttft_mean"])
#         csv_row("ttft_min_s",         e2b["ttft_min"],     e4b["ttft_min"])
#         csv_row("ttft_max_s",         e2b["ttft_max"],     e4b["ttft_max"])
#         csv_row("latency_mean_s",     e2b["lat_mean"],     e4b["lat_mean"])
#         csv_row("latency_min_s",      e2b["lat_min"],      e4b["lat_min"])
#         csv_row("latency_max_s",      e2b["lat_max"],      e4b["lat_max"])
#         csv_row("tps_mean",           e2b["tps_mean"],     e4b["tps_mean"],     lib=False)
#         csv_row("tps_min",            e2b["tps_min"],      e4b["tps_min"],      lib=False)
#         csv_row("tps_max",            e2b["tps_max"],      e4b["tps_max"],      lib=False)
#         csv_row("tokens_mean",        e2b["tokens_mean"],  e4b["tokens_mean"],  lib=False)
#         csv_row("load_ram_gb",        e2b["load_ram_gb"],  e4b["load_ram_gb"])
#         csv_row("load_vram_gb",       e2b["load_vram_gb"], e4b["load_vram_gb"])
#         csv_row("peak_ram_gb",        e2b["peak_ram_gb"],  e4b["peak_ram_gb"])
#         csv_row("peak_vram_gb",       e2b["peak_vram_gb"], e4b["peak_vram_gb"])
#         csv_row("float32_kv_mb",      e2b["float32_kv_mb"],  e4b["float32_kv_mb"])
#         csv_row("turbo_kv_mb",        e2b["turbo_kv_mb"],    e4b["turbo_kv_mb"])
#         csv_row("compression_ratio",  e2b["compression_ratio"], e4b["compression_ratio"], lib=False)

#         # Per-run raw data
#         f.write("\n--- E2B raw runs ---\n")
#         f.write("run,ttft_s,latency_s,tps,tokens\n")
#         for i, r in enumerate(e2b["run_results"], 1):
#             f.write(f"{i},{r['ttft']},{r['latency']},{r['throughput_tps']},{r['tokens_generated']}\n")

#         f.write("\n--- E4B raw runs ---\n")
#         f.write("run,ttft_s,latency_s,tps,tokens\n")
#         for i, r in enumerate(e4b["run_results"], 1):
#             f.write(f"{i},{r['ttft']},{r['latency']},{r['throughput_tps']},{r['tokens_generated']}\n")

#     print(f"  💾  Results saved → {csv_path}\n")


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(
#         description="Comparative GPU benchmark: Gemma 4 E2B vs E4B"
#     )
#     parser.add_argument("--model-e2b",  default=MODEL_SPECS["E2B"][1],
#                         help="Path to E2B model directory")
#     parser.add_argument("--model-e4b",  default=MODEL_SPECS["E4B"][1],
#                         help="Path to E4B model directory")
#     parser.add_argument("--runs",  "-r", type=int, default=DEFAULT_RUNS)
#     parser.add_argument("--prompt","-p", type=str, default=DEFAULT_PROMPT)
#     parser.add_argument("--key-bits",   type=int,  default=DEFAULT_KEY_BITS)
#     parser.add_argument("--val-bits",   type=int,  default=DEFAULT_VAL_BITS)
#     parser.add_argument("--skip-e2b",   action="store_true",
#                         help="Skip E2B and only run E4B (for quick re-runs)")
#     parser.add_argument("--skip-e4b",   action="store_true",
#                         help="Skip E4B and only run E2B")
#     args = parser.parse_args()

#     for label, path in [("E2B", args.model_e2b), ("E4B", args.model_e4b)]:
#         if not os.path.isdir(path):
#             sys.exit(f"❌  {label} model directory not found: {path}")

#     if not torch.cuda.is_available():
#         sys.exit("❌  No CUDA GPU detected. Please run on a machine with a GPU.")

#     device = torch.device("cuda:0")

#     print(SEP2)
#     print("  Gemma 4 E2B vs E4B  ·  TurboQuant KV Cache  ·  Comparative GPU Benchmark")
#     print(SEP2)
#     print(f"  GPU            : {torch.cuda.get_device_name(device)}")
#     print(f"  Key bits       : {args.key_bits}  ({2**args.key_bits} centroids)")
#     print(f"  Value bits     : {args.val_bits}  ({2**args.val_bits} centroids)")
#     print(f"  Context window : {CONTEXT_WINDOW} tokens")
#     print(f"  Max new tokens : {MAX_NEW_TOKENS}")
#     print(f"  Runs per model : {args.runs}")
#     print(f"  Prompt         : \"{args.prompt[:72]}{'…' if len(args.prompt) > 72 else ''}\"")
#     print(SEP2)

#     # Build prompt text once (same for both models)
#     # Use a simple tokenizer-agnostic template; each model's tokenizer
#     # will be tried for apply_chat_template first.
#     prompt_text = args.prompt

#     def build_text(tokenizer):
#         try:
#             return tokenizer.apply_chat_template(
#                 [{"role": "user", "content": prompt_text}],
#                 tokenize=False, add_generation_prompt=True,
#             )
#         except (ValueError, AttributeError):
#             return GEMMA4_CHAT_TEMPLATE.format(prompt=prompt_text)

#     # ── Run E2B ───────────────────────────────────────────────────────────────
#     e2b_results = None
#     if not args.skip_e2b:
#         # Temporarily load tokenizer just to build text, then re-use in benchmark
#         tok_e2b   = AutoTokenizer.from_pretrained(
#             args.model_e2b, trust_remote_code=True, local_files_only=True)
#         text_e2b  = build_text(tok_e2b)
#         del tok_e2b
#         e2b_results = benchmark_model("E2B", args.model_e2b, args, device, text_e2b)

#     # ── Run E4B ───────────────────────────────────────────────────────────────
#     e4b_results = None
#     if not args.skip_e4b:
#         tok_e4b   = AutoTokenizer.from_pretrained(
#             args.model_e4b, trust_remote_code=True, local_files_only=True)
#         text_e4b  = build_text(tok_e4b)
#         del tok_e4b
#         e4b_results = benchmark_model("E4B", args.model_e4b, args, device, text_e4b)

#     # ── Report ────────────────────────────────────────────────────────────────
#     if e2b_results and e4b_results:
#         print_comparison(e2b_results, e4b_results, args)
#         save_csv(e2b_results, e4b_results, args)
#     elif e2b_results:
#         print("\n  (E4B skipped — no comparison produced)")
#     elif e4b_results:
#         print("\n  (E2B skipped — no comparison produced)")


# if __name__ == "__main__":
#     main()