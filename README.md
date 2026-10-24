<div align="center">

# SafeWork CV

**Computer vision for workplace safety: detect falls, edge proximity, immobility, running, lost workers and fire on ordinary CCTV streams.**

[![CI](https://github.com/OWNER/safework-cv/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/safework-cv/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#status-and-safety-notice)

<img src="images/simple.png" alt="SafeWork CV tracking people on a video stream" width="820">

</div>

SafeWork CV reads a video file, an RTSP/HTTP camera stream or a webcam, tracks every person in the frame and raises structured events when something looks wrong. It is built for construction sites, factories, warehouses and other high-risk environments where a worker who falls, collapses or disappears has to be noticed quickly.

It runs fully on your own hardware: no cloud service, no accounts, no data leaves the machine unless you configure a webhook.

## Status and safety notice

SafeWork CV is **alpha software**. The detection rules are heuristics with sensible defaults, not a certified safety system, and they have not been validated on a labelled benchmark. Treat the output as an additional signal for people responsible for safety and never as a replacement for guardrails, supervision, emergency procedures or applicable regulations. Tune and test every rule on footage from your own site before relying on it.

## Features

| Event | What triggers it | Notes |
|---|---|---|
| `FALL` | A person goes from upright to lying within about two seconds, with a fast downward movement of the body centre | Confirmed over 0.4 s to avoid reacting to crouching or bending |
| `IMMOBILE` | A person stays in a lying posture without moving for 8 s | Works from the skeleton when available, falls back to the bounding box |
| `EDGE` | Feet come within 0.3 body lengths of an edge you marked | Edges are drawn once per camera with the built-in editor |
| `RUNNING` | Ground speed above 1.4 body lengths per second for 0.6 s | Speed is measured relative to body size and compensates for perspective |
| `OCCLUDED` | A confirmed worker vanishes in the middle of the frame (behind a column, a wall, a machine) and does not come back for 12 s | The last known point is marked on the frame. Workers who leave through the frame border are not reported |
| `FIRE`, `SMOKE` | A YOLO model with fire/smoke classes, or a colour-and-flicker heuristic for fire when no model is supplied | See [Fire and smoke](#fire-and-smoke) |

Every threshold is configurable and is expressed in body lengths and seconds rather than pixels, so the same configuration works across camera distances and resolutions.

Other highlights:

- **Lying people are not lost.** A standard person detector often misses someone on the ground. A rotated detection pass recovers them, both across the whole frame and around workers who just disappeared.
- **Stable under flickering detections.** Tracks must be confirmed before they can raise alerts, duplicate tracks are merged, and alerts have hysteresis, so a single noisy frame does not produce an event.
- **Correct time on any source.** Video files use the video clock, so results are identical whether the machine processes 5 or 50 frames per second. Live streams use the wall clock and always analyse the newest frame instead of building up latency.
- **Production-friendly I/O.** JSON Lines event log, webhook delivery, annotated alert snapshots, annotated video output, automatic stream reconnects, credentials masked in logs, graceful shutdown on SIGTERM.

## Screenshots

| | |
|:---:|:---:|
| <img src="images/lying%28dangerous%29.png" alt="Person lying on the ground"> | <img src="images/running.png" alt="Running"> |
| **Person lying on the ground** | **Running** |
| <img src="images/lost.png" alt="Lost worker"> | <img src="images/fireexample.png" alt="Fire detection"> |
| **Lost worker, last known point** | **Fire** |
| <img src="images/simple2.png" alt="People tracked in the frame"> | |
| **Tracking in the frame** | |

Overlay legend: green box, tracked person; red, active alert; orange, running; yellow cross, last known point of a lost worker; thin box with `held`, position kept without a fresh detection; cyan line, marked edge. `v=` shows speed in body lengths per second.

## How it works

```mermaid
flowchart LR
    S["Video source<br/>file, RTSP, webcam"] --> D["YOLO person detector<br/>plus rotated pass for lying people"]
    D --> T["Tracker<br/>confirmation, re-linking"]
    T --> P["MediaPipe pose<br/>on person crops"]
    T --> ST["Person state<br/>posture, speed, motion, last known point"]
    P --> ST
    Z["Edges<br/>zones.json"] --> R["Rule engine"]
    ST --> R
    H["Hazard monitor<br/>fire, smoke"] --> R
    R --> E["Events<br/>log, JSONL, webhook, snapshots, video"]
```

1. **Detection.** YOLO finds people on every frame. Every 1.2 s, and around workers who just disappeared, the frame is also scanned rotated by 90 degrees in both directions so that people lying on the ground look upright to the detector.
2. **Tracking.** A lightweight tracker matches detections to tracks. A track must be seen repeatedly before it is trusted, and tracks that are almost entirely covered by another person are merged.
3. **Pose.** MediaPipe estimates a skeleton on the crop of every tracked person a few times per second. Between estimates the skeleton follows the bounding box. Rules never depend on the skeleton alone: if it disappears, bounding-box geometry takes over.
4. **State and rules.** For each person the system keeps posture, ground speed, time since last movement, and the last known position. The rule engine turns that into alerts, and an event tracker turns alerts into `start` and `end` events.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/OWNER/safework-cv.git
cd safework-cv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install .
```

Optional extras:

```bash
pip install ".[screen]"            # screen-capture test mode
pip install -e ".[dev]"            # tests and linting
```

The first run downloads the YOLO weights (`yolo11n.pt`) and the MediaPipe pose model. The pose model is cached in `~/.cache/safework_cv/models` (override with `SAFEWORK_CV_HOME` or `model_dir` in the config).

For GPU inference install a CUDA build of PyTorch first, then pass `--device cuda:0`.

### Docker

```bash
docker build -t safework-cv .
docker run --rm \
  -v "$PWD:/data" \
  -v safework-models:/home/safework/.cache/safework_cv \
  safework-cv run --source site.mp4 --events events.jsonl --snapshots snapshots
```

The container works in `/data`, so relative paths refer to the mounted directory, and it runs as an unprivileged user (uid 1000) that must be able to write there. The named volume keeps the downloaded pose model between runs; the YOLO weights are downloaded into the working directory on first use. The image uses CPU PyTorch. For GPU, build on top of a CUDA base image.

## Quick start

Analyse a video file and watch the result:

```bash
safework-cv run --source site.mp4 --display
```

Press `q` in the window to stop.

Mark the edges where a fall from height is possible, then use them:

```bash
safework-cv zones --source site.mp4 --output zones.json
safework-cv run --source site.mp4 --zones zones.json --display
```

Monitor a camera around the clock, headless, with an event log, alert snapshots and a webhook:

```bash
safework-cv run \
  --source "rtsp://user:password@192.168.1.20:554/stream1" \
  --zones zones.json \
  --config site.yaml \
  --events /var/lib/safework-cv/events.jsonl \
  --snapshots /var/lib/safework-cv/snapshots \
  --webhook https://example.com/hooks/safety
```

Render an annotated video, for example for a demo or a review:

```bash
safework-cv run --source site.mp4 --save-video out/annotated.mp4 --snapshots out/snapshots
```

## Command line reference

Global options go before the command: `safework-cv --log-level DEBUG run ...`.

### `safework-cv run`

| Option | Description |
|---|---|
| `--source` | Required. Video file, `rtsp://` or `http://` URL, camera index (`0`), or `screen[:monitor]` |
| `--config FILE` | YAML file with configuration overrides |
| `--zones FILE` | Edges created with `safework-cv zones` |
| `--display` | Show the annotated video window (`q` quits) |
| `--display-scale X` | Scale of the window |
| `--events FILE` | Append events to this JSON Lines file |
| `--webhook URL` | POST every event as JSON to an `http(s)` URL |
| `--snapshots DIR` | Save an annotated JPEG whenever an alert starts |
| `--save-video FILE` | Write the annotated video |
| `--max-frames N` | Stop after N processed frames |
| `--stride N` | Process every Nth frame of a video file |
| `--region L,T,W,H` | Screen source only: capture this rectangle |
| `--select-region` | Screen source only: pick the rectangle with the mouse |
| `--weights FILE` | Person detector weights (default `yolo11n.pt`) |
| `--hazard-weights FILE` | YOLO weights with fire and smoke classes |
| `--no-color-fire` | Disable the colour-based fire heuristic |
| `--no-pose` | Disable skeleton estimation (faster, bounding-box rules only) |
| `--device` | Inference device: `cpu`, `cuda:0`, `mps` |

### `safework-cv zones`

Opens one frame of the source and lets you draw edges. Click to add points, `Enter` to finish an edge, `u` to undo a point, `d` to delete the last edge, `c` to clear everything, `s` to save and exit, `q` to quit without saving.

### `safework-cv config`

Prints the default configuration as YAML. Redirect it to a file to start your own: `safework-cv config > site.yaml`.

Exit codes: `0` on success or graceful stop, `2` on configuration or source errors, `1` when the zone editor was closed without saving.

## Edges

An edge is a polyline along an unprotected edge, an open shaft, a missing railing or any line that a worker should not approach. Coordinates are stored normalised to the frame, so a `zones.json` keeps working if the camera resolution changes:

```json
{
  "version": 1,
  "edges": [
    [[0.12, 0.64], [0.48, 0.61], [0.83, 0.66]]
  ]
}
```

Each camera needs its own file. If the camera moves, redraw the edges.

## Configuration

Run `safework-cv config` for the complete list of parameters or read [`configs/default.yaml`](configs/default.yaml). A configuration file only needs the keys you want to change:

```yaml
camera_up: [0.0, -1.0]
run_speed: 1.8
immobile_seconds: 15
occlusion_seconds: 20
pose_variant: full
```

Unknown keys are rejected, so typos do not pass silently.

| Parameter | Default | Meaning |
|---|---|---|
| `camera_up` | `[0, -1]` | Image-space vector that points up in the scene. Tilt it for cameras that are rolled, for example `[0.2, -1.0]` |
| `run_speed` | `1.4` | Running threshold in body lengths per second |
| `immobile_seconds` | `8` | Lying without motion before `IMMOBILE` |
| `occlusion_seconds` | `12` | Time out of sight before `OCCLUDED` |
| `edge_distance_ratio` | `0.3` | Distance from feet to an edge, in body lengths, that raises `EDGE` |
| `fall_min_drop_ratio` | `0.25` | Minimum downward movement of the body centre, in body sizes, for a fall |
| `fall_min_speed` | `0.5` | Minimum speed of that movement, in body sizes per second |
| `yolo_weights` | `yolo11n.pt` | Person detector. Larger models (`yolo11s.pt`, `yolo11m.pt`) find small or partly hidden people more reliably |
| `yolo_conf` | `0.15` | Detector threshold. Low on purpose: weak detections can extend existing tracks but never start new ones |
| `enable_pose` | `true` | Use MediaPipe skeletons |
| `pose_variant` | `lite` | `lite`, `full` or `heavy`. Heavier models are more accurate and slower |
| `color_fire` | `true` | Colour-and-flicker fire heuristic when no hazard weights are given |
| `alert_clear_seconds` | `2` | How long an alert must be absent before its `end` event |

### Tuning on your footage

- Look at `v=` above each person. Walking usually reads around 0.5 to 1.0 and running above 1.5. Set `run_speed` between the two for your camera angle.
- `held` next to a person means the detector did not see them on the latest frames and the box is being kept. If it stays on permanently, switch to a larger model, raise `yolo_imgsz`, or improve the camera view.
- The `fps` counter in the corner is the processing speed. Below roughly five frames per second, speed and fall estimates become unreliable. Use `--no-pose`, a smaller `yolo_imgsz`, a GPU, or the camera's lower-resolution sub-stream.
- Mount the camera so that people appear mostly upright and from the side or from above at an angle. Rules for posture assume the "up" direction of the scene is roughly vertical in the image.

## Events

Every alert produces a `start` event, and an `end` event when it clears. Events are logged, appended to `--events` and posted to `--webhook` as one JSON object each:

```json
{
  "phase": "start",
  "kind": "FALL",
  "track_id": 3,
  "text": "sudden fall",
  "point": [640.2, 515.7],
  "video_time": 3.9,
  "wall_time": "2026-10-23T04:12:09.411+00:00",
  "duration": null,
  "frame": 39,
  "snapshot": "snapshots/00000039_fall_3.jpg"
}
```

| Field | Description |
|---|---|
| `phase` | `start` or `end` |
| `kind` | `FALL`, `IMMOBILE`, `EDGE`, `RUNNING`, `OCCLUDED`, `FIRE`, `SMOKE` |
| `track_id` | Person identifier within the run, `null` for hazards |
| `point` | Pixel position of the person or hazard |
| `video_time` | Seconds since the first processed frame |
| `wall_time` | UTC time when the event was emitted |
| `duration` | Length of the alert, set on `end` events |
| `frame` | Source frame index |
| `snapshot` | Annotated JPEG saved at the start of the alert, if `--snapshots` is set |

Webhook delivery runs in a background thread with a small queue, so a slow or unavailable endpoint never blocks video processing. Failed deliveries are logged and not retried.

## Fire and smoke

Standard person-detection weights do not know about fire. There are two modes:

- **Model (recommended).** Pass `--hazard-weights` with a YOLO model trained on fire and smoke. Any model whose class names contain `fire`, `flame` or `smoke` works.
- **Heuristic fallback.** Without weights, fire is detected from flame-coloured pixels that flicker over about 1.5 s. People are masked out so skin and clothing are ignored. It detects flames only, not smoke, and it can be fooled by welding, flames on a monitor or flashing orange lamps. Disable it with `--no-color-fire` if it causes false alarms.

## Deployment

Recommendations for permanent installations:

- Use the camera's **sub-stream** if it offers a lower resolution; detection quality rarely needs more than 720p.
- RTSP is read over TCP by default. The reader keeps only the newest frame, so a slow machine drops frames instead of accumulating delay. Interrupted streams reconnect automatically every `reconnect_seconds`.
- Run one process per camera, supervised by systemd, Docker or your orchestrator.

Example systemd unit:

```ini
[Unit]
Description=SafeWork CV camera 1
After=network-online.target

[Service]
ExecStart=/opt/safework-cv/.venv/bin/safework-cv run \
  --source rtsp://user:password@192.168.1.20:554/stream1 \
  --zones /etc/safework-cv/cam1-zones.json \
  --config /etc/safework-cv/site.yaml \
  --events /var/lib/safework-cv/cam1-events.jsonl \
  --snapshots /var/lib/safework-cv/cam1-snapshots
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

SIGTERM stops the process cleanly and finishes pending files.

### Privacy and data protection

Snapshots and annotated videos contain images of people. Store them on encrypted disks with restricted access, define a retention period, and make sure the monitoring complies with local labour and data-protection law (for example GDPR). Stream credentials are masked in logs, but they are visible in the process list when passed on the command line; prefer a read-only camera account.

## Screen-capture test mode

Production use reads files and camera streams. For quick experiments it is also possible to analyse whatever is playing on your own screen, for example a video in a browser:

```bash
pip install ".[screen]"
safework-cv run --source screen --select-region --display
safework-cv run --source screen:2 --region 0,0,1280,720 --display
```

`--select-region` lets you drag a rectangle over the video. Keep the preview window outside the captured area, otherwise the program analyses its own output. This mode is a convenience for testing and is not intended for deployments.

## Limitations

- All reasoning is 2D from a single camera. Falls straight towards or away from the camera, and cameras mounted directly overhead, are weak spots for posture-based rules.
- The default COCO-trained person detector can miss people who are heavily occluded, very small or lying in unusual poses. The rotated pass and track keeping reduce this but cannot remove it.
- Identity across long occlusions is based on proximity only, without appearance matching. In crowds two workers can swap identities.
- Only the rules listed above exist today. There is no PPE detection, no danger-zone polygons yet and no multi-camera fusion.
- The thresholds are starting values chosen by reasoning, not by evaluation on labelled data. Expect to tune them.

## Roadmap

- Polygonal danger zones with entry and dwell-time rules
- Additional incident scenarios
- Pluggable detector backends, including permissively licensed ones
- Evaluation tooling and a labelled benchmark for the rules
- ONNX and TensorRT export for edge devices

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

The tests exercise the whole analysis pipeline with scripted detections and synthetic video, so they run in seconds and need neither a GPU nor model downloads.

```
src/safework_cv/
  cli.py            command line interface
  config.py         configuration schema and YAML loading
  sources.py        file, stream and screen sources
  perception.py     YOLO person detection and MediaPipe pose
  tracker.py        detection-to-track association
  state.py          per-person posture, speed, motion and fall logic
  rules.py          alert rules
  hazards.py        fire and smoke
  alerts.py         alerts and start/end events
  sinks.py          log, JSON Lines and webhook outputs
  analyzer.py       frame-by-frame pipeline
  runner.py         source loop, display, snapshots and video output
  zones.py          edge storage
  zone_editor.py    interactive edge editor
  overlay.py        drawing
tests/              unit, scenario and integration tests
configs/            default configuration
images/             screenshots shown in this README
.github/            CI workflow and issue templates
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Security issues should be reported as described in [SECURITY.md](SECURITY.md).

## License

SafeWork CV is released under the [GNU Affero General Public License v3.0 or later](LICENSE).

The default person detector is built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is itself licensed under AGPL-3.0. If you modify SafeWork CV and offer it to others over a network, the AGPL requires you to make the corresponding source available to those users. Organisations that cannot comply with the AGPL should obtain a commercial licence from Ultralytics or replace the detector.

| Component | Licence |
|---|---|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 |
| [OpenCV](https://opencv.org/) | Apache-2.0 |
| [NumPy](https://numpy.org/) | BSD-3-Clause |
| [PyYAML](https://pyyaml.org/) | MIT |
| [mss](https://github.com/BoboTiG/python-mss) (optional) | MIT |
