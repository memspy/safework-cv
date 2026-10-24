from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

import yaml

from . import __version__
from .analyzer import Analyzer
from .config import Config
from .hazards import HazardMonitor
from .runner import Runner
from .sinks import JsonlSink, LogSink, MultiSink, WebhookSink
from .sources import ScreenSource, SourceError, open_source, parse_region
from .zones import ZoneSet

log = logging.getLogger("safework_cv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safework-cv",
        description="Computer vision safety monitoring for construction sites and factories.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="analyze a video source and emit safety events")
    _add_source_arguments(run)
    run.add_argument("--config", type=Path, help="YAML file with configuration overrides")
    run.add_argument("--zones", type=Path, help="JSON file with fall-risk edges (see 'zones')")
    run.add_argument("--display", action="store_true", help="show the annotated video window")
    run.add_argument("--display-scale", type=float, default=1.0, help="window scale factor")
    run.add_argument("--events", type=Path, help="append events to this JSON Lines file")
    run.add_argument("--webhook", help="POST every event as JSON to this http(s) URL")
    run.add_argument("--snapshots", type=Path, help="directory for annotated alert snapshots")
    run.add_argument("--save-video", type=Path, help="write the annotated video to this file")
    run.add_argument("--max-frames", type=int, help="stop after this many processed frames")
    run.add_argument("--weights", help="person detector weights (default: yolo11n.pt)")
    run.add_argument("--hazard-weights", help="YOLO weights with fire/smoke classes")
    run.add_argument(
        "--no-color-fire", action="store_true", help="disable the colour fire heuristic"
    )
    run.add_argument("--no-pose", action="store_true", help="disable skeleton estimation")
    run.add_argument("--device", help="inference device: cpu, cuda:0, mps (default: auto)")

    zones = sub.add_parser("zones", help="draw fall-risk edges on a frame and save them")
    _add_source_arguments(zones)
    zones.add_argument(
        "--output", type=Path, default=Path("zones.json"), help="zones file to write"
    )

    sub.add_parser("config", help="print the default configuration as YAML")
    return parser


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        required=True,
        help="video file, rtsp:// or http:// URL, camera index, or 'screen[:monitor]'",
    )
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame of a file")
    parser.add_argument("--region", help="screen source only: left,top,width,height")
    parser.add_argument(
        "--select-region",
        action="store_true",
        help="screen source only: pick the region with the mouse",
    )


def _open(args, cfg: Config):
    region = parse_region(args.region) if args.region else None
    source = open_source(
        args.source, stride=args.stride, region=region, reconnect_seconds=cfg.reconnect_seconds
    )
    if args.select_region:
        if not isinstance(source, ScreenSource):
            source.close()
            raise SourceError("--select-region works only with the screen source")
        source.pick_region()
    return source


def _build_sink(args):
    sinks = [LogSink()]
    if args.events:
        sinks.append(JsonlSink(args.events))
    if args.webhook:
        sinks.append(WebhookSink(args.webhook))
    return MultiSink(sinks)


def _build_analyzer(cfg: Config, zones: ZoneSet) -> Analyzer:
    from .perception import PersonDetector, PoseEstimator

    detector = PersonDetector(cfg)
    pose = PoseEstimator(cfg) if cfg.enable_pose else None
    return Analyzer(cfg, detector, pose, HazardMonitor(cfg), zones)


def _command_run(args) -> int:
    cfg = Config.from_file(args.config) if args.config else Config()
    cfg = cfg.with_overrides(
        yolo_weights=args.weights,
        hazard_weights=args.hazard_weights,
        device=args.device,
        color_fire=False if args.no_color_fire else None,
        enable_pose=False if args.no_pose else None,
    )
    zones = ZoneSet.load(args.zones) if args.zones else ZoneSet()
    sink = _build_sink(args)
    source = None
    try:
        source = _open(args, cfg)
        runner = Runner(
            _build_analyzer(cfg, zones),
            sink,
            display=args.display,
            display_scale=args.display_scale,
            snapshot_dir=args.snapshots,
            video_path=args.save_video,
        )
        processed = runner.run(source, args.max_frames)
    finally:
        if source is not None:
            source.close()
        sink.close()
    log.info("finished, %d frames processed", processed)
    return 0


def _command_zones(args) -> int:
    from .zone_editor import ZoneEditor

    source = _open(args, Config())
    try:
        frame = source.read()
    finally:
        source.close()
    if frame is None:
        raise SourceError("could not read a frame from the source")
    zones = ZoneSet.load(args.output) if args.output.exists() else ZoneSet()
    result = ZoneEditor(frame.image, zones).run()
    if result is None:
        log.info("nothing saved")
        return 1
    result.save(args.output)
    log.info("saved %d edge(s) to %s", len(result.edges), args.output)
    return 0


def _command_config(args) -> int:
    sys.stdout.write(yaml.safe_dump(Config().to_mapping(), sort_keys=False))
    return 0


COMMANDS = {"run": _command_run, "zones": _command_zones, "config": _command_config}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        log.info("stopped")
        return 0
    except (SourceError, ValueError, OSError, ImportError) as exc:
        log.error("%s", exc)
        log.debug("details", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
