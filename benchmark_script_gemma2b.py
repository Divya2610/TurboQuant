#!/usr/bin/env python3
"""
benchmark.py — Gemma 4 E2B · TurboQuant KV Cache · CPU-only
=============================================================
Metrics  :  TTFT  |  Latency  |  Throughput  |  RAM (GB)
Context  :  2 048 tokens (fixed)
Backend  :  CPU only, float32
Model    :  google/gemma-4-E2B  (local path)

Fixes applied vs original:
  1. KV cache MB formula now includes num_kv_heads × head_dim
     (absolute sizes were drastically understated before; compression
      ratio was still correct because the same factor was missing from
      both sides of the division)
  2. uint8 overflow guard added — bucketize indices are stored as uint8
     which silently overflows above 8 bits (> 255 levels)
  3. Codebook is fitted on synthetic Gaussian data (not real KV tensors).
     This is a known approximation: post-rotation KV values are roughly
     Gaussian so it works well in practice, but a calibration pass on
     real data would give a tighter codebook.
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

SEP  = "─" * 70
SEP2 = "═" * 70


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMICCACHE HELPERS  — version-agnostic read/write
#
# Old API (transformers < ~4.47):  cache.key_cache / cache.value_cache  (lists)
# New API (transformers >= ~4.47):  cache.layers  (list of CacheLayer objects)
#                                   cache.update(k, v, layer_idx) to populate
#
# We detect which API is present at runtime and use it accordingly.
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
    else:
        # New API: cache.layers is a list of CacheLayer objects
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
# TURBOQUANT CORE
# ─────────────────────────────────────────────────────────────────────────────

def build_rotation_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    G = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q


def fit_lloyd_max(n_bits: int, n_samples: int = 100_000, n_iter: int = 150) -> tuple:
    """
    Fits an optimal scalar quantizer for a Gaussian distribution using the
    Lloyd-Max algorithm.

    NOTE: The codebook is fitted on synthetic torch.randn() samples, not on
    real KV tensors extracted from the model.  Post-rotation KV activations
    are empirically close to Gaussian, so this approximation works well in
    practice.  For the tightest possible codebook, replace the sample tensor
    here with KV vectors collected during a short calibration forward pass.
    """
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
        self.k_bounds = None
        self.k_cents  = None
        self.v_bounds = None
        self.v_cents  = None

    def _compress(self, x, Q, bounds):
        y       = x @ Q.T
        scale   = y.std(dim=-1, keepdim=True).clamp(min=1e-8)
        indices = torch.bucketize((y / scale).contiguous(), bounds)
        # FIX 2: indices are stored as uint8 (range 0-255).
        # This is safe for up to 8 bits (256 levels). The bit-width guard in
        # TurboQuantEngine.__init__ ensures we never exceed that limit.
        return indices.to(torch.uint8), scale

    def _decompress(self, indices, scale, Q, cents):
        return (cents[indices.long()] * scale) @ Q

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

        # ── FIX 2: uint8 overflow guard ──────────────────────────────────────
        # Compressed indices are stored as torch.uint8 (0-255).
        # Any bit width above 8 would produce more than 256 levels and
        # silently overflow, corrupting the stored indices.
        if key_bits > 8 or val_bits > 8:
            raise ValueError(
                f"key_bits={key_bits} and val_bits={val_bits} must both be "
                f"≤ 8 because indices are stored as uint8 (max 256 levels). "
                f"Use key_bits ≤ 8 and val_bits ≤ 8."
            )
        if key_bits < 1 or val_bits < 1:
            raise ValueError("key_bits and val_bits must both be ≥ 1.")
        # ─────────────────────────────────────────────────────────────────────

        cfg      = model.config
        text_cfg = getattr(cfg, "text_config", cfg)   # unwrap Gemma4Config

        num_layers    = text_cfg.num_hidden_layers
        self.key_bits = key_bits
        self.val_bits = val_bits

        # head_dim: size of each attention head vector
        head_dim = getattr(text_cfg, "head_dim", None)
        if head_dim is None:
            head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
        self.head_dim = head_dim

        # ── FIX 1: store num_kv_heads so the compression formula is correct ──
        # KV projections use num_key_value_heads (GQA/MQA), not num_attention_heads.
        # Without this, the printed MB figures are wrong by a factor of
        # (num_kv_heads × head_dim).
        self.num_kv_heads = getattr(text_cfg, "num_key_value_heads",
                                    text_cfg.num_attention_heads)
        self.num_layers   = num_layers
        # ─────────────────────────────────────────────────────────────────────

        print(f"\n  [TurboQuant] Fitting Lloyd-Max codebooks …")
        print(f"              Key bits  : {key_bits}  →  {2**key_bits} centroids")
        print(f"              Value bits: {val_bits}  →  {2**val_bits} centroids")

        t0 = time.perf_counter()
        k_bounds, k_cents = fit_lloyd_max(key_bits)
        v_bounds, v_cents = fit_lloyd_max(val_bits)
        print(f"              Done in {time.perf_counter() - t0:.2f}s")

        self.compressors = []
        for i in range(num_layers):
            lc          = LayerCompressor(head_dim, key_bits, val_bits, layer_idx=i)
            lc.k_bounds = k_bounds
            lc.k_cents  = k_cents
            lc.v_bounds = v_bounds
            lc.v_cents  = v_cents
            self.compressors.append(lc)

        print(f"  [TurboQuant] {num_layers} layer compressors ready "
              f"(head_dim={head_dim}, num_kv_heads={self.num_kv_heads})")

    def compress_cache(self, cache: DynamicCache) -> list:
        """Extract and compress KV tensors from cache. Returns list of dicts or None per layer."""
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
            if data is not None:
                layer_kvs.append(cmp.decompress(data))
            else:
                layer_kvs.append(None)
        return build_cache_from_kv(layer_kvs)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE BENCHMARK RUN
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_single(model, tokenizer, input_ids, attention_mask, tq_engine) -> dict:
    eos_id           = tokenizer.eos_token_id
    tokens_generated = 0
    t_start          = time.perf_counter()

    # PREFILL — feed all input tokens in one forward pass
    out              = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    ttft             = time.perf_counter() - t_start
    next_token       = out.logits[:, -1:, :].argmax(dim=-1)
    tokens_generated += 1

    compressed_store = tq_engine.compress_cache(out.past_key_values)
    del out
    gc.collect()

    decode_mask = torch.ones((1, input_ids.shape[-1] + 1), dtype=attention_mask.dtype)

    # DECODE LOOP — one token at a time, compress/decompress cache each step
    for _ in range(MAX_NEW_TOKENS - 1):
        if next_token.item() == eos_id:
            break

        past_kv = tq_engine.decompress_to_cache(compressed_store)
        out     = model(
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
# OUTPUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_header(args):
    print(SEP2)
    print("  Gemma 4 E2B  ·  TurboQuant KV Cache  ·  CPU-only Benchmark")
    print(SEP2)
    print(f"  Model          : {args.model}")
    print(f"  Key bits       : {args.key_bits}  (Lloyd-Max + rotation, {2**args.key_bits} levels)")
    print(f"  Value bits     : {args.val_bits}  (Lloyd-Max + rotation, {2**args.val_bits} levels)")
    print(f"  Context window : {CONTEXT_WINDOW} tokens (fixed)")
    print(f"  Max new tokens : {MAX_NEW_TOKENS}")
    print(f"  Runs           : {args.runs}")
    print(f"  Device         : CPU  (torch.float32, no GPU layers)")
    print(SEP2)


def print_run_row(idx, r, run_ram):
    print(
        f"  Run {idx + 1:>2}  │"
        f"  TTFT {r['ttft']:>7.3f}s"
        f"  │  Latency {r['latency']:>8.3f}s"
        f"  │  TPS {r['throughput_tps']:>7.3f}"
        f"  │  Tokens {r['tokens_generated']:>4}"
        f"  │  RAM {run_ram:.2f} GB"
    )


def print_summary(results, peak_ram, args, tq_engine):
    ttfts     = [r["ttft"]           for r in results]
    latencies = [r["latency"]        for r in results]
    tpss      = [r["throughput_tps"] for r in results]

    def _sd(vals):
        return f"{statistics.stdev(vals):.4f}" if len(vals) > 1 else "      —"

    # ── FIX 1: correct KV cache size formula ─────────────────────────────────
    # Full formula:
    #   tokens × 2 (K+V) × layers × num_kv_heads × head_dim × bytes_per_element
    #
    # Original code was missing (num_kv_heads × head_dim), which caused the
    # displayed MB figures to be off by that factor (e.g. ~2000× too small for
    # a model with num_kv_heads=8, head_dim=256).
    #
    # The compression RATIO was still correct in the original because the same
    # missing factor appeared in both the float32 and TurboQuant formulas and
    # cancelled out in the division.
    num_kv_heads  = tq_engine.num_kv_heads
    head_dim      = tq_engine.head_dim
    num_layers    = tq_engine.num_layers

    float32_kv_mb = (
        CONTEXT_WINDOW * 2 * num_layers * num_kv_heads * head_dim * 4
        / (1024 ** 2)
    )
    turbo_kv_mb = (
        CONTEXT_WINDOW * 2 * num_layers * num_kv_heads * head_dim
        * (args.key_bits + args.val_bits) / (8 * 2)          # bits → bytes, /2 for K and V separately averaged
        / (1024 ** 2)
    )

    # Cleaner re-expression to make the intent explicit:
    #   float32: each element = 4 bytes (32 bits)
    #   turbo  : key elements = key_bits/8 bytes, value elements = val_bits/8 bytes
    #   total turbo bytes = tokens × layers × num_kv_heads × head_dim × (key_bits + val_bits) / 8
    turbo_kv_mb = (
        CONTEXT_WINDOW * num_layers * num_kv_heads * head_dim
        * (args.key_bits + args.val_bits)
        / 8                    # bits → bytes
        / (1024 ** 2)          # bytes → MB
    )
    # ─────────────────────────────────────────────────────────────────────────

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
    print(SEP)
    print()
    print(f"  TURBOQUANT KV CACHE COMPRESSION")
    print(f"  (context={CONTEXT_WINDOW} tokens · {num_layers} layers · "
          f"{num_kv_heads} kv_heads · head_dim={head_dim})")
    print(SEP)
    print(f"  Float32 KV cache size  : {float32_kv_mb:.1f} MB")
    print(f"  TurboQuant KV size     : {turbo_kv_mb:.1f} MB")
    print(f"  Compression ratio      : {float32_kv_mb / turbo_kv_mb:.1f}×")
    print(SEP2)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    "-m", default="./google-gemma-4-E2B")
    parser.add_argument("--runs",     "-r", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--prompt",   "-p", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--key-bits",       type=int, default=DEFAULT_KEY_BITS)
    parser.add_argument("--val-bits",       type=int, default=DEFAULT_VAL_BITS)
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        sys.exit(f"❌  Model directory not found: {args.model}")

    print_header(args)

    global_ram = RAMSampler().start()

    print("\n  Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
    )

    print("  Loading model on CPU (this may take a few minutes) …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype             = torch.float32,
        device_map        = "cpu",
        trust_remote_code = True,
        local_files_only  = True,
    )
    model.eval()
    print(f"  ✅  Model loaded  |  RAM: {global_ram.current_gb:.2f} GB\n")

    # FIX 2: TurboQuantEngine now validates key_bits / val_bits ≤ 8 on init
    # and raises a clear ValueError before any computation if violated.
    tq_engine = TurboQuantEngine(model, args.key_bits, args.val_bits)

    print(f"\n  Prompt: \"{args.prompt[:80]}{'…' if len(args.prompt) > 80 else ''}\"")

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
    attention_mask = inputs.get("attention_mask",
                                torch.ones_like(input_ids))[:, :CONTEXT_WINDOW]

    print(f"  Input tokens : {input_ids.shape[-1]}")
    print(f"\n  {SEP}")
    print(f"  {'Run':>5}  {'TTFT':>10}  {'Latency':>11}  {'TPS':>10}  {'Tokens':>8}  {'RAM':>8}")
    print(f"  {SEP}")

    results = []
    for i in range(args.runs):
        run_ram  = RAMSampler().start()
        result   = run_single(model, tokenizer, input_ids, attention_mask, tq_engine)
        peak_run = run_ram.stop()
        results.append(result)
        print_run_row(i, result, peak_run)

    peak_total = global_ram.stop()

    # FIX 1: pass tq_engine into print_summary so it can read
    # num_kv_heads, head_dim, and num_layers for the correct MB formula.
    print_summary(results, peak_total, args, tq_engine)

    csv_path = "benchmark_results.csv"
    with open(csv_path, "w") as f:
        f.write("run,ttft_s,latency_s,throughput_tps,tokens_generated\n")
        for i, r in enumerate(results, 1):
            f.write(f"{i},{r['ttft']},{r['latency']},{r['throughput_tps']},{r['tokens_generated']}\n")
        f.write(f"\npeak_ram_gb,{peak_total:.3f}\n")

    print(f"  💾  Results saved → {csv_path}\n")


if __name__ == "__main__":
    main()




# #!/usr/bin/env python3
# """
# benchmark.py — Gemma 4 E2B · TurboQuant KV Cache · CPU-only
# =============================================================
# Metrics  :  TTFT  |  Latency  |  Throughput  |  RAM (GB)
# Context  :  2 048 tokens (fixed)
# Backend  :  CPU only, float32
# Model    :  google/gemma-4-E2B  (local path)
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
#     def __init__(self, model, key_bits: int, val_bits: int):
#         cfg      = model.config
#         text_cfg = getattr(cfg, "text_config", cfg)   # unwrap Gemma4Config

#         num_layers    = text_cfg.num_hidden_layers
#         self.key_bits = key_bits
#         self.val_bits = val_bits

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
#         print(f"              Done in {time.perf_counter() - t0:.2f}s")

#         self.compressors = []
#         for i in range(num_layers):
#             lc          = LayerCompressor(head_dim, key_bits, val_bits, layer_idx=i)
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
# def run_single(model, tokenizer, input_ids, attention_mask, tq_engine) -> dict:
#     eos_id           = tokenizer.eos_token_id
#     tokens_generated = 0
#     t_start          = time.perf_counter()

#     # PREFILL
#     out              = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
#     ttft             = time.perf_counter() - t_start
#     next_token       = out.logits[:, -1:, :].argmax(dim=-1)
#     tokens_generated += 1

#     compressed_store = tq_engine.compress_cache(out.past_key_values)
#     del out
#     gc.collect()

#     decode_mask = torch.ones((1, input_ids.shape[-1] + 1), dtype=attention_mask.dtype)

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
# # OUTPUT HELPERS
# # ─────────────────────────────────────────────────────────────────────────────

# def print_header(args):
#     print(SEP2)
#     print("  Gemma 4 E2B  ·  TurboQuant KV Cache  ·  CPU-only Benchmark")
#     print(SEP2)
#     print(f"  Model          : {args.model}")
#     print(f"  Key bits       : {args.key_bits}  (Lloyd-Max + rotation, {2**args.key_bits} levels)")
#     print(f"  Value bits     : {args.val_bits}  (Lloyd-Max + rotation, {2**args.val_bits} levels)")
#     print(f"  Context window : {CONTEXT_WINDOW} tokens (fixed)")
#     print(f"  Max new tokens : {MAX_NEW_TOKENS}")
#     print(f"  Runs           : {args.runs}")
#     print(f"  Device         : CPU  (torch.float32, no GPU layers)")
#     print(SEP2)


# def print_run_row(idx, r, run_ram):
#     print(
#         f"  Run {idx + 1:>2}  │"
#         f"  TTFT {r['ttft']:>7.3f}s"
#         f"  │  Latency {r['latency']:>8.3f}s"
#         f"  │  TPS {r['throughput_tps']:>7.3f}"
#         f"  │  Tokens {r['tokens_generated']:>4}"
#         f"  │  RAM {run_ram:.2f} GB"
#     )


# def print_summary(results, peak_ram, args):
#     ttfts     = [r["ttft"]           for r in results]
#     latencies = [r["latency"]        for r in results]
#     tpss      = [r["throughput_tps"] for r in results]

#     def _sd(vals):
#         return f"{statistics.stdev(vals):.4f}" if len(vals) > 1 else "      —"

#     float32_kv_mb = CONTEXT_WINDOW * 2 * 35 * 4 / (1024 ** 2)
#     turbo_kv_mb   = CONTEXT_WINDOW * 35 * (args.key_bits + args.val_bits) / 8 / (1024 ** 2)

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
#     print(SEP)
#     print()
#     print("  TURBOQUANT KV CACHE COMPRESSION  (at 2 048-token context, 35 layers)")
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
#     parser.add_argument("--model",    "-m", default="./google-gemma-4-E2B")
#     parser.add_argument("--runs",     "-r", type=int, default=DEFAULT_RUNS)
#     parser.add_argument("--prompt",   "-p", type=str, default=DEFAULT_PROMPT)
#     parser.add_argument("--key-bits",       type=int, default=DEFAULT_KEY_BITS)
#     parser.add_argument("--val-bits",       type=int, default=DEFAULT_VAL_BITS)
#     args = parser.parse_args()

#     if not os.path.isdir(args.model):
#         sys.exit(f"❌  Model directory not found: {args.model}")

#     print_header(args)

#     global_ram = RAMSampler().start()

#     print("\n  Loading tokenizer …")
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.model, trust_remote_code=True, local_files_only=True,
#     )

#     print("  Loading model on CPU (this may take a few minutes) …")
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model,
#         dtype             = torch.float32,
#         device_map        = "cpu",
#         trust_remote_code = True,
#         local_files_only  = True,
#     )
#     model.eval()
#     print(f"  ✅  Model loaded  |  RAM: {global_ram.current_gb:.2f} GB\n")

#     tq_engine = TurboQuantEngine(model, args.key_bits, args.val_bits)

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
#     print(f"  {'Run':>5}  {'TTFT':>10}  {'Latency':>11}  {'TPS':>10}  {'Tokens':>8}  {'RAM':>8}")
#     print(f"  {SEP}")

#     results = []
#     for i in range(args.runs):
#         run_ram  = RAMSampler().start()
#         result   = run_single(model, tokenizer, input_ids, attention_mask, tq_engine)
#         peak_run = run_ram.stop()
#         results.append(result)
#         print_run_row(i, result, peak_run)

#     peak_total = global_ram.stop()
#     print_summary(results, peak_total, args)

#     csv_path = "benchmark_results.csv"
#     with open(csv_path, "w") as f:
#         f.write("run,ttft_s,latency_s,throughput_tps,tokens_generated\n")
#         for i, r in enumerate(results, 1):
#             f.write(f"{i},{r['ttft']},{r['latency']},{r['throughput_tps']},{r['tokens_generated']}\n")
#         f.write(f"\npeak_ram_gb,{peak_total:.3f}\n")

#     print(f"  💾  Results saved → {csv_path}\n")


# if __name__ == "__main__":
#     main()