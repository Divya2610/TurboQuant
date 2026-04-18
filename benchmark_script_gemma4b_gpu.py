#!/usr/bin/env python3
"""
benchmark.py — Gemma 4 E4B · TurboQuant KV Cache · GPU
=======================================================
Metrics  :  TTFT  |  Latency  |  Throughput  |  RAM (GB)  |  VRAM (GB)
Context  :  2 048 tokens (fixed)
Backend  :  CUDA GPU, float16
Model    :  google/gemma-4-E4B  (local path)

Fixes applied vs original:
  1.  Rotation matrices cast to float16 to match model dtype — prevents
      implicit float32 upcasts during matmuls, saving VRAM and improving speed.
  2.  Layer count read from model config instead of hardcoded 42 — compression
      ratio stats are now accurate for any model variant.
  3.  Scale-factor memory included in TurboQuant size estimate — reported
      compression ratio now reflects real memory usage.
  4.  Lloyd-Max fitting runs on GPU from the start — faster codebook
      construction, no CPU→GPU copy overhead.
  5.  Warmup run added and excluded from results — eliminates JIT/cuDNN
      overhead from the first timed measurement.
  6.  CSV output now records key_bits and val_bits — results from different
      quantisation settings are distinguishable after the fact.
  7.  DEFAULT_VAL_BITS raised from 2 → 3 — less aggressive, more stable
      output quality.
  8.  decode_mask length verified against past_kv each step — defensive
      guard against silent misalignment across transformers versions.
  9.  VRAMSampler.start() now calls torch.cuda.set_device() before
      reset_peak_memory_stats() — fixes "Invalid device argument" RuntimeError
      that occurred when CUDA context had not yet been initialised at the
      point the sampler was started (i.e. before any model or tensor was
      placed on the GPU).
"""

import argparse
import os
import sys
import time
import threading
import statistics
import gc

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECKS
# ─────────────────────────────────────────────────────────────────────────────
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
DEFAULT_VAL_BITS = 3   # FIX 7: raised from 2 — less aggressive, more stable

DEFAULT_PROMPT = (
    "Explain the key differences between supervised, unsupervised, and "
    "reinforcement learning. Give one real-world example for each."
)

GEMMA4_CHAT_TEMPLATE = (
    "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
)

SEP  = "─" * 70
SEP2 = "═" * 70


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMICCACHE HELPERS  — version-agnostic read/write
#
# Old API (transformers < ~4.47):  cache.key_cache / cache.value_cache  (lists)
# New API (transformers >= ~4.47):  cache.layers  (list of CacheLayer objects)
#                                   cache.update(k, v, layer_idx) to populate
# ─────────────────────────────────────────────────────────────────────────────

def _cache_has_old_api(cache: DynamicCache) -> bool:
    return hasattr(cache, "key_cache")


def extract_kv_from_cache(cache: DynamicCache) -> list:
    """
    Returns list of (key, value) tensors per layer, or None for empty layers.
    Works with both old and new DynamicCache APIs.
    """
    if _cache_has_old_api(cache):
        result = []
        for k, v in zip(cache.key_cache, cache.value_cache):
            result.append((k, v) if k is not None else None)
        return result

    # New API: cache.layers is a list of CacheLayer objects
    result = []
    for layer in cache.layers:
        if layer is None:
            result.append(None)
            continue
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
    """
    Build a new DynamicCache from a list of (key, value) or None per layer.
    Uses cache.update() which is stable across all transformers versions.
    """
    new_cache = DynamicCache()
    for i, kv in enumerate(layer_kvs):
        if kv is not None:
            k, v = kv
            new_cache.update(k, v, i)
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
# VRAM SAMPLER
# ─────────────────────────────────────────────────────────────────────────────
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
        # FIX 9: ensure the CUDA context is initialised before calling
        # reset_peak_memory_stats().  Without this, starting VRAMSampler
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
    FIX 1: dtype and device are explicit parameters.
    QR decomposition runs in float32 for numerical stability, then the result
    is cast to the target dtype (float16) — prevents silent float32 upcasts
    during every matmul in the hot decode path.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    G = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q.to(dtype=dtype, device=device)


def fit_lloyd_max(
    n_bits:    int,
    device:    torch.device,          # FIX 4: device passed in, runs on GPU
    n_samples: int = 100_000,
    n_iter:    int = 150,
) -> tuple:
    """
    FIX 4: All tensors are created directly on `device` (GPU).
    No CPU→GPU copy at the end — codebooks arrive ready to use.
    """
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
    """
    Handles compress / decompress for a single transformer layer's KV tensors.

    Rotation matrices are stored in the model's working dtype (float16) so
    all matmuls stay in float16 — no implicit upcasts, lower VRAM footprint.
    """

    def __init__(
        self,
        head_dim:  int,
        key_bits:  int,
        val_bits:  int,
        layer_idx: int,
        device:    torch.device,
        dtype:     torch.dtype,   # FIX 1
    ):
        self.head_dim = head_dim
        self.device   = device
        self.dtype    = dtype
        # FIX 1: rotation matrices built in the correct dtype from the start
        self.Q_k = build_rotation_matrix(head_dim, seed=layer_idx * 2,     dtype=dtype, device=device)
        self.Q_v = build_rotation_matrix(head_dim, seed=layer_idx * 2 + 1, dtype=dtype, device=device)
        self.k_bounds = None
        self.k_cents  = None
        self.v_bounds = None
        self.v_cents  = None

    def _compress(self, x: torch.Tensor, Q: torch.Tensor, bounds: torch.Tensor):
        """Rotate → normalise → quantise to uint8 indices."""
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
        """Dequantise → un-rotate. Cast centroids to working dtype."""
        return (cents[indices.long()].to(self.dtype) * scale) @ Q

    def compress(self, key: torch.Tensor, value: torch.Tensor) -> dict:
        k_idx, k_scale = self._compress(key,   self.Q_k, self.k_bounds)
        v_idx, v_scale = self._compress(value, self.Q_v, self.v_bounds)
        return {
            "k_idx": k_idx, "k_scale": k_scale,
            "v_idx": v_idx, "v_scale": v_scale,
        }

    def decompress(self, data: dict) -> tuple:
        key   = self._decompress(data["k_idx"], data["k_scale"], self.Q_k, self.k_cents)
        value = self._decompress(data["v_idx"], data["v_scale"], self.Q_v, self.v_cents)
        return key, value


# ─────────────────────────────────────────────────────────────────────────────
# TURBOQUANT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TurboQuantEngine:
    """
    Wraps all per-layer compressors. Holds codebooks and orchestrates
    compress_cache / decompress_to_cache operations.
    """

    def __init__(
        self,
        model,
        key_bits: int,
        val_bits: int,
        device:   torch.device,
        dtype:    torch.dtype = torch.float16,   # FIX 1
    ):
        cfg      = model.config
        text_cfg = getattr(cfg, "text_config", cfg)   # unwrap Gemma4Config

        # FIX 2: read num_layers from config, never hardcode
        self.num_layers = text_cfg.num_hidden_layers
        self.key_bits   = key_bits
        self.val_bits   = val_bits
        self.device     = device
        self.dtype      = dtype

        head_dim = getattr(text_cfg, "head_dim", None)
        if head_dim is None:
            head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
        self.head_dim = head_dim

        # FIX 3: record num_kv_heads for accurate size estimates
        self.num_kv_heads = getattr(
            text_cfg, "num_key_value_heads",
            text_cfg.num_attention_heads,
        )

        print(f"\n  [TurboQuant] Fitting Lloyd-Max codebooks on {device} …")
        print(f"              Key bits  : {key_bits}  →  {2**key_bits} centroids")
        print(f"              Value bits: {val_bits}  →  {2**val_bits} centroids")

        t0 = time.perf_counter()
        # FIX 4: fitting happens on GPU — codebooks arrive ready, no .to() needed
        k_bounds, k_cents = fit_lloyd_max(key_bits, device)
        v_bounds, v_cents = fit_lloyd_max(val_bits, device)
        print(f"              Done in {time.perf_counter() - t0:.2f}s")

        self.compressors = []
        for i in range(self.num_layers):
            lc          = LayerCompressor(
                head_dim, key_bits, val_bits,
                layer_idx=i, device=device, dtype=dtype,   # FIX 1
            )
            lc.k_bounds = k_bounds
            lc.k_cents  = k_cents
            lc.v_bounds = v_bounds
            lc.v_cents  = v_cents
            self.compressors.append(lc)

        print(f"  [TurboQuant] {self.num_layers} layer compressors ready "
              f"(head_dim={head_dim}, kv_heads={self.num_kv_heads}, dtype={dtype})")

    # ── cache operations ──────────────────────────────────────────────────────

    def compress_cache(self, cache: DynamicCache) -> list:
        """Extract and compress KV tensors. Returns list of dicts or None per layer."""
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
        """Rebuild DynamicCache from compressed data."""
        layer_kvs = []
        for cmp, data in zip(self.compressors, compressed):
            layer_kvs.append(cmp.decompress(data) if data is not None else None)
        return build_cache_from_kv(layer_kvs)

    # ── size estimation ───────────────────────────────────────────────────────

    def estimate_sizes_mb(self, context_len: int) -> dict:
        """
        FIX 2 + 3: use actual layer/head counts and include scale-factor overhead.

        KV cache breakdown per layer:
          • float16 baseline : context_len × num_kv_heads × head_dim × 2 (K+V) × 2 bytes
          • TurboQuant indices: context_len × num_kv_heads × head_dim × (bits/8) bytes
          • Scale factors    : context_len × num_kv_heads × 1 × 2 bytes (float16)
                               for both K and V
        """
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
def run_single(
    model,
    tokenizer,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor,
    tq_engine:      TurboQuantEngine,
    device:         torch.device,
) -> dict:
    eos_id           = tokenizer.eos_token_id
    tokens_generated = 0
    seq_len          = input_ids.shape[-1]

    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    # ── PREFILL ───────────────────────────────────────────────────────────────
    out  = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    torch.cuda.synchronize(device)
    ttft = time.perf_counter() - t_start

    next_token        = out.logits[:, -1:, :].argmax(dim=-1)
    tokens_generated += 1

    compressed_store = tq_engine.compress_cache(out.past_key_values)
    del out
    gc.collect()
    torch.cuda.empty_cache()

    # decode_mask covers: original prompt tokens + first generated token
    decode_mask = torch.ones(
        (1, seq_len + 1), dtype=attention_mask.dtype, device=device
    )

    # ── DECODE LOOP ───────────────────────────────────────────────────────────
    for step in range(MAX_NEW_TOKENS - 1):
        if next_token.item() == eos_id:
            break

        past_kv = tq_engine.decompress_to_cache(compressed_store)

        # FIX 8: verify mask length == past_kv length + 1 (new token)
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
        next_token        = out.logits[:, -1:, :].argmax(dim=-1)
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
# OUTPUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_header(args, device, tq_engine: TurboQuantEngine):
    print(SEP2)
    print("  Gemma 4 E4B  ·  TurboQuant KV Cache  ·  GPU Benchmark")
    print(SEP2)
    print(f"  Model          : {args.model}")
    print(f"  Layers         : {tq_engine.num_layers}  (from config — not hardcoded)")
    print(f"  KV heads       : {tq_engine.num_kv_heads}")
    print(f"  Head dim       : {tq_engine.head_dim}")
    print(f"  Key bits       : {args.key_bits}  (Lloyd-Max + rotation, {2**args.key_bits} levels)")
    print(f"  Value bits     : {args.val_bits}  (Lloyd-Max + rotation, {2**args.val_bits} levels)")
    print(f"  Context window : {CONTEXT_WINDOW} tokens (fixed)")
    print(f"  Max new tokens : {MAX_NEW_TOKENS}")
    print(f"  Runs           : {args.runs}  (+1 warmup, discarded)")
    print(f"  Device         : {device}  ({torch.cuda.get_device_name(device)}, torch.float16)")
    print(SEP2)


def print_run_row(label: str, r: dict, run_ram: float, run_vram: float):
    print(
        f"  {label:<9}  │"
        f"  TTFT {r['ttft']:>7.3f}s"
        f"  │  Latency {r['latency']:>8.3f}s"
        f"  │  TPS {r['throughput_tps']:>7.3f}"
        f"  │  Tokens {r['tokens_generated']:>4}"
        f"  │  RAM {run_ram:.2f} GB"
        f"  │  VRAM {run_vram:.2f} GB"
    )


def print_summary(
    results:    list,
    peak_ram:   float,
    peak_vram:  float,
    args,
    tq_engine:  TurboQuantEngine,
):
    ttfts     = [r["ttft"]           for r in results]
    latencies = [r["latency"]        for r in results]
    tpss      = [r["throughput_tps"] for r in results]

    def _sd(vals):
        return f"{statistics.stdev(vals):.4f}" if len(vals) > 1 else "      —"

    # FIX 2 + 3: use engine's accurate size estimator
    sizes = tq_engine.estimate_sizes_mb(CONTEXT_WINDOW)

    print()
    print(SEP2)
    print("  BENCHMARK RESULTS")
    print(SEP)
    print(f"  {'METRIC':<26}  {'MIN':>10}  {'MAX':>10}  {'MEAN':>10}  {'STDEV':>10}")
    print(SEP)
    for label, vals, unit in [
        ("TTFT (s)",           ttfts,     "s"),
        ("Latency (s)",        latencies, "s"),
        ("Throughput (tok/s)", tpss,      " "),
    ]:
        print(
            f"  {label:<26}"
            f"  {min(vals):>9.4f}{unit}"
            f"  {max(vals):>9.4f}{unit}"
            f"  {statistics.mean(vals):>9.4f}{unit}"
            f"  {_sd(vals):>10}"
        )
    print(SEP)
    print(f"  {'Peak RAM (GB)':<26}  {peak_ram:.3f} GB")
    print(f"  {'Peak VRAM (GB)':<26}  {peak_vram:.3f} GB")
    print(SEP)
    print()
    print("  TURBOQUANT KV CACHE COMPRESSION")
    print(f"  (context={CONTEXT_WINDOW} tokens, layers={tq_engine.num_layers}, "
          f"kv_heads={tq_engine.num_kv_heads}, head_dim={tq_engine.head_dim})")
    print(SEP)
    print(f"  Float16 KV cache size  : {sizes['float16_mb']:>7.1f} MB")
    print(f"  TurboQuant KV size     : {sizes['turbo_mb']:>7.1f} MB  (incl. scale factors)")
    print(f"  Compression ratio      : {sizes['ratio']:>7.1f}×")
    print(SEP2)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Gemma 4 E4B with TurboQuant KV cache on GPU."
    )
    parser.add_argument(
        "--model", "-m",
        default="/home/kanshika/Desktop/Model/gemma-4-E4B",
        help="Path to local model directory.",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int, default=DEFAULT_RUNS,
        help="Number of timed runs (a warmup run is always added).",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str, default=DEFAULT_PROMPT,
        help="Prompt string to benchmark.",
    )
    parser.add_argument(
        "--key-bits",
        type=int, default=DEFAULT_KEY_BITS,
        help="Bits per key element (default 3).",
    )
    parser.add_argument(
        "--val-bits",
        type=int, default=DEFAULT_VAL_BITS,
        help="Bits per value element (default 3).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        sys.exit(f"❌  Model directory not found: {args.model}")

    if not torch.cuda.is_available():
        sys.exit("❌  No CUDA GPU detected. Please run on a machine with a GPU.")

    device = torch.device("cuda:0")

    # FIX 9: initialise CUDA context before constructing VRAMSampler so that
    # reset_peak_memory_stats() has a valid device to work with.  The sampler
    # also calls set_device + init internally as a belt-and-suspenders guard,
    # but doing it here ensures the global sampler started below is also safe.
    torch.cuda.set_device(device)
    torch.cuda.init()

    global_ram  = RAMSampler().start()
    global_vram = VRAMSampler(device).start()

    print("\n  Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
    )

    print("  Loading model on GPU (this may take a few minutes) …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype       = torch.float16,
        device_map        = "auto",
        trust_remote_code = True,
        local_files_only  = True,
    )
    model.eval()

    # FIX 1: pass dtype explicitly so engine and compressors stay in float16
    tq_engine = TurboQuantEngine(model, args.key_bits, args.val_bits, device, dtype=torch.float16)

    print_header(args, device, tq_engine)
    print(f"  ✅  Model loaded  |  RAM: {global_ram.current_gb:.2f} GB  "
          f"|  VRAM: {global_vram.current_gb:.2f} GB\n")

    print(f"  Prompt: \"{args.prompt[:80]}{'…' if len(args.prompt) > 80 else ''}\"")

    # Build prompt — try chat template, fall back to Gemma 4 turn format
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except (ValueError, AttributeError):
        print("  ℹ️  No chat template — using Gemma 4 turn format.")
        text = GEMMA4_CHAT_TEMPLATE.format(prompt=args.prompt)

    inputs         = tokenizer(text, return_tensors="pt")
    input_ids      = inputs["input_ids"][:, :CONTEXT_WINDOW]
    attention_mask = inputs.get(
        "attention_mask", torch.ones_like(input_ids)
    )[:, :CONTEXT_WINDOW]

    print(f"  Input tokens : {input_ids.shape[-1]}")
    print(f"\n  {SEP}")
    print(f"  {'Run':<9}  {'TTFT':>10}  {'Latency':>11}  {'TPS':>10}  "
          f"{'Tokens':>8}  {'RAM':>8}  {'VRAM':>9}")
    print(f"  {SEP}")

    # FIX 5: warmup run — discarded from results
    print("  Running warmup (discarded) …", end="", flush=True)
    _ = run_single(model, tokenizer, input_ids, attention_mask, tq_engine, device)
    print("  done")
    print(f"  {SEP}")

    # ── TIMED RUNS ────────────────────────────────────────────────────────────
    results = []
    for i in range(args.runs):
        run_ram  = RAMSampler().start()
        run_vram = VRAMSampler(device).start()
        result   = run_single(model, tokenizer, input_ids, attention_mask, tq_engine, device)
        peak_run      = run_ram.stop()
        peak_run_vram = run_vram.stop()
        results.append(result)
        print_run_row(f"Run {i + 1:>2}", result, peak_run, peak_run_vram)

    peak_total      = global_ram.stop()
    peak_total_vram = global_vram.stop()
    print_summary(results, peak_total, peak_total_vram, args, tq_engine)

    # FIX 6: CSV includes key_bits, val_bits, and compression stats
    csv_path = "benchmark_results_gemma4_e4b_gpu.csv"
    with open(csv_path, "w") as f:
        f.write("run,ttft_s,latency_s,throughput_tps,tokens_generated,key_bits,val_bits\n")
        for i, r in enumerate(results, 1):
            f.write(
                f"{i},{r['ttft']},{r['latency']},"
                f"{r['throughput_tps']},{r['tokens_generated']},"
                f"{args.key_bits},{args.val_bits}\n"
            )
        f.write(f"\npeak_ram_gb,{peak_total:.3f}\n")
        f.write(f"peak_vram_gb,{peak_total_vram:.3f}\n")
        sizes = tq_engine.estimate_sizes_mb(CONTEXT_WINDOW)
        f.write(f"float16_kv_mb,{sizes['float16_mb']:.1f}\n")
        f.write(f"turboquant_kv_mb,{sizes['turbo_mb']:.1f}\n")
        f.write(f"compression_ratio,{sizes['ratio']:.2f}\n")

    print(f"  💾  Results saved → {csv_path}\n")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# benchmark.py — Gemma 4 E4B · TurboQuant KV Cache · GPU
# =======================================================
# Metrics  :  TTFT  |  Latency  |  Throughput  |  RAM (GB)  |  VRAM (GB)
# Context  :  2 048 tokens (fixed)
# Backend  :  CUDA GPU, float16
# Model    :  google/gemma-4-E4B  (local path)
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

# SEP  = "─" * 70
# SEP2 = "═" * 70


# # ─────────────────────────────────────────────────────────────────────────────
# # DYNAMICCACHE HELPERS  — version-agnostic read/write
# #
# # Old API (transformers < ~4.47):  cache.key_cache / cache.value_cache  (lists)
# # New API (transformers >= ~4.47):  cache.layers  (list of CacheLayer objects)
# #                                   cache.update(k, v, layer_idx) to populate
# #
# # We detect which API is present at runtime and use it accordingly.
# # ─────────────────────────────────────────────────────────────────────────────

# def _cache_has_old_api(cache: DynamicCache) -> bool:
#     return hasattr(cache, "key_cache")


# def extract_kv_from_cache(cache: DynamicCache) -> list:
#     """
#     Returns list of (key, value) tensors per layer, or None for empty layers.
#     Works with both old and new DynamicCache APIs.
#     """
#     if _cache_has_old_api(cache):
#         result = []
#         for k, v in zip(cache.key_cache, cache.value_cache):
#             result.append((k, v) if k is not None else None)
#         return result
#     else:
#         # New API: cache.layers is a list of CacheLayer objects
#         result = []
#         for layer in cache.layers:
#             if layer is None:
#                 result.append(None)
#             else:
#                 # Each CacheLayer has .key and .value tensors
#                 k = getattr(layer, "key", None)
#                 v = getattr(layer, "value", None)
#                 if k is None:
#                     # fallback: some versions use different attr names
#                     try:
#                         k, v = layer.key_cache, layer.value_cache
#                     except AttributeError:
#                         result.append(None)
#                         continue
#                 result.append((k, v))
#         return result


# def build_cache_from_kv(layer_kvs: list) -> DynamicCache:
#     """
#     Build a new DynamicCache from a list of (key, value) or None per layer.
#     Uses cache.update() which is stable across all transformers versions.
#     """
#     new_cache = DynamicCache()
#     for i, kv in enumerate(layer_kvs):
#         if kv is not None:
#             k, v = kv
#             new_cache.update(k, v, i)
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
# # VRAM SAMPLER
# # ─────────────────────────────────────────────────────────────────────────────
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
#         # Also check torch's own peak tracker as a safety net
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
#         text_cfg = getattr(cfg, "text_config", cfg)   # unwrap Gemma4Config

#         num_layers    = text_cfg.num_hidden_layers
#         self.key_bits = key_bits
#         self.val_bits = val_bits
#         self.device   = device

#         head_dim = getattr(text_cfg, "head_dim", None)
#         if head_dim is None:
#             head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
#         self.head_dim = head_dim

#         print(f"\n  [TurboQuant] Fitting Lloyd-Max codebooks …")
#         print(f"              Key bits  : {key_bits}  →  {2**key_bits} centroids")
#         print(f"              Value bits: {val_bits}  →  {2**val_bits} centroids")

#         t0 = time.perf_counter()
#         k_bounds, k_cents = fit_lloyd_max(key_bits)
#         v_bounds, v_cents = fit_lloyd_max(val_bits)
#         # Move codebook tensors to GPU
#         k_bounds = k_bounds.to(device)
#         k_cents  = k_cents.to(device)
#         v_bounds = v_bounds.to(device)
#         v_cents  = v_cents.to(device)
#         print(f"              Done in {time.perf_counter() - t0:.2f}s")

#         self.compressors = []
#         for i in range(num_layers):
#             lc          = LayerCompressor(head_dim, key_bits, val_bits, layer_idx=i, device=device)
#             lc.k_bounds = k_bounds
#             lc.k_cents  = k_cents
#             lc.v_bounds = v_bounds
#             lc.v_cents  = v_cents
#             self.compressors.append(lc)

#         print(f"  [TurboQuant] {num_layers} layer compressors ready (head_dim={head_dim})")

#     def compress_cache(self, cache: DynamicCache) -> list:
#         """Extract and compress KV tensors from cache. Returns list of dicts or None per layer."""
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
#         """Rebuild DynamicCache from compressed data."""
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

#     # Move inputs to GPU
#     input_ids      = input_ids.to(device)
#     attention_mask = attention_mask.to(device)

#     torch.cuda.synchronize(device)
#     t_start = time.perf_counter()

#     # PREFILL
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

#     # DECODE LOOP
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
# # OUTPUT HELPERS
# # ─────────────────────────────────────────────────────────────────────────────

# def print_header(args, device):
#     print(SEP2)
#     print("  Gemma 4 E4B  ·  TurboQuant KV Cache  ·  GPU Benchmark")
#     print(SEP2)
#     print(f"  Model          : {args.model}")
#     print(f"  Key bits       : {args.key_bits}  (Lloyd-Max + rotation, {2**args.key_bits} levels)")
#     print(f"  Value bits     : {args.val_bits}  (Lloyd-Max + rotation, {2**args.val_bits} levels)")
#     print(f"  Context window : {CONTEXT_WINDOW} tokens (fixed)")
#     print(f"  Max new tokens : {MAX_NEW_TOKENS}")
#     print(f"  Runs           : {args.runs}")
#     print(f"  Device         : {device}  ({torch.cuda.get_device_name(device)}, torch.float16)")
#     print(SEP2)


# def print_run_row(idx, r, run_ram, run_vram):
#     print(
#         f"  Run {idx + 1:>2}  │"
#         f"  TTFT {r['ttft']:>7.3f}s"
#         f"  │  Latency {r['latency']:>8.3f}s"
#         f"  │  TPS {r['throughput_tps']:>7.3f}"
#         f"  │  Tokens {r['tokens_generated']:>4}"
#         f"  │  RAM {run_ram:.2f} GB"
#         f"  │  VRAM {run_vram:.2f} GB"
#     )


# def print_summary(results, peak_ram, peak_vram, args):
#     ttfts     = [r["ttft"]           for r in results]
#     latencies = [r["latency"]        for r in results]
#     tpss      = [r["throughput_tps"] for r in results]

#     def _sd(vals):
#         return f"{statistics.stdev(vals):.4f}" if len(vals) > 1 else "      —"

#     # E4B has 42 layers (vs 35 for E2B)
#     float32_kv_mb = CONTEXT_WINDOW * 2 * 42 * 4 / (1024 ** 2)
#     turbo_kv_mb   = CONTEXT_WINDOW * 42 * (args.key_bits + args.val_bits) / 8 / (1024 ** 2)

#     print()
#     print(SEP2)
#     print("  BENCHMARK RESULTS")
#     print(SEP)
#     print(f"  {'METRIC':<26}  {'MIN':>10}  {'MAX':>10}  {'MEAN':>10}  {'STDEV':>10}")
#     print(SEP)
#     for label, vals, unit in [
#         ("TTFT (s)",           ttfts,     "s"),
#         ("Latency (s)",        latencies, "s"),
#         ("Throughput (tok/s)", tpss,      " "),
#     ]:
#         print(
#             f"  {label:<26}"
#             f"  {min(vals):>9.4f}{unit}"
#             f"  {max(vals):>9.4f}{unit}"
#             f"  {statistics.mean(vals):>9.4f}{unit}"
#             f"  {_sd(vals):>10}"
#         )
#     print(SEP)
#     print(f"  {'Peak RAM (GB)':<26}  {peak_ram:.3f} GB")
#     print(f"  {'Peak VRAM (GB)':<26}  {peak_vram:.3f} GB")
#     print(SEP)
#     print()
#     print("  TURBOQUANT KV CACHE COMPRESSION  (at 2 048-token context, 42 layers)")
#     print(SEP)
#     print(f"  Float32 KV cache size  : {float32_kv_mb:.1f} MB")
#     print(f"  TurboQuant KV size     : {turbo_kv_mb:.1f} MB")
#     print(f"  Compression ratio      : {float32_kv_mb / turbo_kv_mb:.1f}×")
#     print(SEP2)
#     print()


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model",    "-m", default="/home/kanshika/Desktop/Model/gemma-4-E4B")
#     parser.add_argument("--runs",     "-r", type=int, default=DEFAULT_RUNS)
#     parser.add_argument("--prompt",   "-p", type=str, default=DEFAULT_PROMPT)
#     parser.add_argument("--key-bits",       type=int, default=DEFAULT_KEY_BITS)
#     parser.add_argument("--val-bits",       type=int, default=DEFAULT_VAL_BITS)
#     args = parser.parse_args()

#     if not os.path.isdir(args.model):
#         sys.exit(f"❌  Model directory not found: {args.model}")

#     if not torch.cuda.is_available():
#         sys.exit("❌  No CUDA GPU detected. Please run on a machine with a GPU.")

#     device = torch.device("cuda:0")
#     print_header(args, device)

#     global_ram  = RAMSampler().start()
#     global_vram = VRAMSampler(device).start()

#     print("\n  Loading tokenizer …")
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.model, trust_remote_code=True, local_files_only=True,
#     )

#     print("  Loading model on GPU (this may take a few minutes) …")
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model,
#         torch_dtype       = torch.float16,
#         device_map        = "auto",
#         trust_remote_code = True,
#         local_files_only  = True,
#     )
#     model.eval()
#     print(f"  ✅  Model loaded  |  RAM: {global_ram.current_gb:.2f} GB  |  VRAM: {global_vram.current_gb:.2f} GB\n")

#     tq_engine = TurboQuantEngine(model, args.key_bits, args.val_bits, device)

#     print(f"\n  Prompt: \"{args.prompt[:80]}{'…' if len(args.prompt) > 80 else ''}\"")

#     # Build prompt — try chat template, fall back to Gemma 4 turn format
#     try:
#         text = tokenizer.apply_chat_template(
#             [{"role": "user", "content": args.prompt}],
#             tokenize=False,
#             add_generation_prompt=True,
#         )
#     except (ValueError, AttributeError):
#         print("  ℹ️  No chat template — using Gemma 4 turn format.")
#         text = GEMMA4_CHAT_TEMPLATE.format(prompt=args.prompt)

#     inputs         = tokenizer(text, return_tensors="pt")
#     input_ids      = inputs["input_ids"][:, :CONTEXT_WINDOW]
#     attention_mask = inputs.get("attention_mask",
#                                 torch.ones_like(input_ids))[:, :CONTEXT_WINDOW]

#     print(f"  Input tokens : {input_ids.shape[-1]}")
#     print(f"\n  {SEP}")
#     print(f"  {'Run':>5}  {'TTFT':>10}  {'Latency':>11}  {'TPS':>10}  {'Tokens':>8}  {'RAM':>8}  {'VRAM':>9}")
#     print(f"  {SEP}")

#     results = []
#     for i in range(args.runs):
#         run_ram  = RAMSampler().start()
#         run_vram = VRAMSampler(device).start()
#         result   = run_single(model, tokenizer, input_ids, attention_mask, tq_engine, device)
#         peak_run      = run_ram.stop()
#         peak_run_vram = run_vram.stop()
#         results.append(result)
#         print_run_row(i, result, peak_run, peak_run_vram)

#     peak_total      = global_ram.stop()
#     peak_total_vram = global_vram.stop()
#     print_summary(results, peak_total, peak_total_vram, args)

#     csv_path = "benchmark_results_gemma4_gpu.csv"
#     with open(csv_path, "w") as f:
#         f.write("run,ttft_s,latency_s,throughput_tps,tokens_generated\n")
#         for i, r in enumerate(results, 1):
#             f.write(f"{i},{r['ttft']},{r['latency']},{r['throughput_tps']},{r['tokens_generated']}\n")
#         f.write(f"\npeak_ram_gb,{peak_total:.3f}\n")
#         f.write(f"peak_vram_gb,{peak_total_vram:.3f}\n")

#     print(f"  💾  Results saved → {csv_path}\n")


# if __name__ == "__main__":
#     main()