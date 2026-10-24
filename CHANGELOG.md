# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

Initial public version.

### Added

- Detection of falls, lying without motion, proximity to marked edges, running, workers lost out of sight, fire and smoke.
- Person detection with YOLO, a rotated detection pass that recovers people lying on the ground, MediaPipe skeletons, and a tracker with confirmation and re-linking.
- Sources: video files, RTSP and HTTP streams with automatic reconnects, webcams, and an optional screen-capture mode for quick tests.
- Outputs: log, JSON Lines event file, webhook, annotated alert snapshots and annotated video.
- `safework-cv run`, `safework-cv zones` for drawing edges, and `safework-cv config` for printing the default configuration.
- YAML configuration with validation of keys and value types.
- Docker image, GitHub Actions workflow and test suite.
