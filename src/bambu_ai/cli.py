"""Command-line interface: bambu-ai <command>."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from .config import Config
from .monitor import Monitor
from .server import serve
from .vision.factory import build_vision_model

DEFAULT_CONFIG = "config.yaml"


def _load_config(path: str) -> Config:
    p = Path(path)
    if not p.exists():
        sys.exit(f"config not found: {path} (copy config.example.yaml to config.yaml)")
    return Config.load(p)


def _setup_logging(cfg: Config) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_camera(cfg: Config):
    if cfg.camera_provider == "bambu":
        from .camera import BambuCamera
        return BambuCamera(cfg.printer_host, cfg.printer_access_code, cfg.camera_port)
    if cfg.camera_provider == "file":
        from .camera import FileCamera
        return FileCamera(cfg.camera_file_dir, interval=1.0 / max(0.05, cfg.inference_fps))
    from .camera import MockCamera
    return MockCamera()


def _build_printer(cfg: Config):
    if cfg.camera_provider == "mock":
        from .printer import MockPrinter
        return MockPrinter()
    from .printer import BambuPrinter
    return BambuPrinter(cfg.printer_host, cfg.printer_serial, cfg.printer_access_code, cfg.printer_mqtt_port)


# -- commands ----------------------------------------------------------------

async def cmd_monitor(cfg: Config) -> None:
    _setup_logging(cfg)
    camera = _build_camera(cfg)
    printer = _build_printer(cfg)
    vision = build_vision_model(cfg)
    monitor = Monitor(cfg, camera, printer, vision)
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(__import__("signal").SIGINT, monitor.stop)
    loop.add_signal_handler(__import__("signal").SIGTERM, monitor.stop)
    # run server in a thread, monitor in main loop
    import threading
    t = threading.Thread(target=serve, args=(monitor, cfg.server_host, cfg.server_port), daemon=True)
    t.start()
    await monitor.run()


async def cmd_status(cfg: Config) -> None:
    printer = _build_printer(cfg)
    await printer.connect()
    try:
        s = await printer.get_status()
        print(f"state={s.state.value} job={s.job_name} progress={s.progress_pct:.0f}%")
    finally:
        await printer.close()


async def cmd_test_camera(cfg: Config) -> None:
    _setup_logging(cfg)
    camera = _build_camera(cfg)
    await camera.connect()
    try:
        f = await camera.get_frame()
        out = Path("test_camera.jpg")
        out.write_bytes(f.data)
        print(f"OK: got frame ({len(f.data)} bytes) -> {out}")
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    finally:
        await camera.close()


async def cmd_test_printer(cfg: Config) -> None:
    printer = _build_printer(cfg)
    await printer.connect()
    try:
        s = await printer.get_status()
        print(f"OK: state={s.state.value} job={s.job_name}")
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    finally:
        await printer.close()


async def cmd_pause(cfg: Config) -> None:
    printer = _build_printer(cfg)
    await printer.connect()
    try:
        await printer.pause("manual via CLI")
        print("pause sent")
    finally:
        await printer.close()


def cmd_benchmark(cfg: Config) -> None:
    from .benchmark import run_benchmark
    run_benchmark(cfg)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bambu-ai")
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("monitor", "status", "test-camera", "test-printer", "pause", "benchmark-model"):
        sp = sub.add_parser(name)
        sp.set_defaults(name=name)
    args = p.parse_args(argv)
    cfg = _load_config(args.config)

    if args.name == "benchmark-model":
        cmd_benchmark(cfg)
        return
    async def run():
        if args.name == "monitor":
            await cmd_monitor(cfg)
        elif args.name == "status":
            await cmd_status(cfg)
        elif args.name == "test-camera":
            await cmd_test_camera(cfg)
        elif args.name == "test-printer":
            await cmd_test_printer(cfg)
        elif args.name == "pause":
            await cmd_pause(cfg)
    asyncio.run(run())


if __name__ == "__main__":
    main()
