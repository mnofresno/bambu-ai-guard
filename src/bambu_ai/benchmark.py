"""Model benchmark: latency / FPS per backend on the current machine.

Usage: bambu-ai benchmark-model
Compares onnxruntime execution providers (coreml / cpu / mps) for the
configured model and reports avg / p95 latency and FPS. Picks the backend
with the best latency / simplicity tradeoff.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np

from .config import Config
from .models import DetectionContext, Frame
from .vision.onnx import OnnxYoloModel


def _sample_frames(cfg: Config, n: int = 20) -> list[Frame]:
    """Use real frames if available, else synthesize noisy images."""
    frames: list[Frame] = []
    d = Path(cfg.camera_file_dir)
    if d.exists():
        for p in sorted(d.glob("*.jpg"))[:n]:
            frames.append(Frame(data=p.read_bytes()))
    while len(frames) < n:
        arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG")
        frames.append(Frame(data=buf.getvalue()))
    return frames[:n]


def _bench_backend(cfg: Config, backend: str, frames: list[Frame]) -> dict:
    model = OnnxYoloModel(
        cfg.vision_model, backend=backend,
        input_size=cfg.vision_input_size, confidence=cfg.vision_confidence,
    )
    import asyncio
    async def run():
        times = []
        ctx = DetectionContext(printer_state=None, elapsed_seconds=0)  # type: ignore
        for f in frames:
            t0 = time.perf_counter()
            await model.analyze(f, ctx)
            times.append((time.perf_counter() - t0) * 1000.0)
        await model.close()
        return times
    times = asyncio.run(run())
    ms = sorted(times)
    p95 = ms[int(0.95 * (len(ms) - 1))]
    avg = statistics.mean(ms)
    rss_mb = _rss_mb()
    return {
        "backend": backend,
        "model": Path(cfg.vision_model).name,
        "resolution": cfg.vision_input_size,
        "ram_mb": rss_mb,
        "avg_ms": round(avg, 1),
        "p95_ms": round(p95, 1),
        "fps": round(1000.0 / avg, 1),
    }


def _rss_mb() -> int:
    try:
        import platform
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KB
        if platform.system() == "Darwin":
            return int(raw / (1024 * 1024))
        return int(raw / 1024)
    except Exception:
        return 0


def run_benchmark(cfg: Config) -> None:
    from .vision.onnx import OnnxYoloModel
    if not Path(cfg.vision_model).exists():
        print(f"model not found: {cfg.vision_model} (download first, see README)")
        return
    frames = _sample_frames(cfg, n=20)
    print(f"benchmarking {cfg.vision_model} on {len(frames)} frames\n")
    results = []
    for backend in ("coreml", "cpu"):
        try:
            r = _bench_backend(cfg, backend, frames)
            results.append(r)
            print(f"  {r['backend']:>7}: avg={r['avg_ms']:>7.1f}ms p95={r['p95_ms']:>7.1f}ms "
                  f"fps={r['fps']:>6.1f} ram={r['ram_mb']}MB")
        except Exception as e:
            print(f"  {backend:>7}: skipped ({e})")
    if not results:
        print("no backend available")
        return
    best = min(results, key=lambda r: r["avg_ms"])
    print(f"\nrecommended backend: {best['backend']} "
          f"(avg {best['avg_ms']}ms, {best['fps']}fps)")
