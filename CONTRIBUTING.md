# Contributing

Thank you for helping to make workplaces safer. Bug reports, footage-based findings, new rules, documentation and tests are all welcome.

## Development setup

```bash
git clone https://github.com/OWNER/safework-cv.git
cd safework-cv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The test suite needs no GPU, no camera and no model downloads. It drives the whole pipeline with scripted detections and synthetic video. If you only want to work on the logic you can skip the heavy dependencies, which is exactly what CI does:

```bash
pip install --no-deps -e .
pip install numpy opencv-python-headless PyYAML pytest ruff
```

## Checks

```bash
ruff check .
ruff format --check .
pytest
```

All three must pass before a pull request can be merged.

## Making a change

- **Keep thresholds relative.** Distances are expressed in body lengths and times in seconds, never in pixels, so that a configuration works across cameras.
- **Test behaviour with scenarios.** Changes to rules, tracking or posture logic need a scenario in `tests/test_scenarios.py`. `tests/helpers.py` builds scripted detections, so a scenario is a short function that returns boxes for a given time.
- **Adding a new alert kind** touches, in this order: the measurement in `state.py`, the alert in `rules.py`, `PRIORITY` in `alerts.py`, the colour in `overlay.py`, the parameters in `config.py` and `configs/default.yaml` (a test keeps them in sync), the README table, and tests.
- **Keep heavy imports lazy.** `ultralytics` and `mediapipe` are imported inside the classes that use them so that everything else can be tested without them.
- **Style.** Code is formatted by `ruff format`. Prefer small functions and descriptive names to comments, and do not add comments that restate the code.
- **No new runtime dependencies** without a good reason. Mention the licence of any dependency you add in the pull request.

## Pull requests

- Keep them focused. One behaviour change per pull request is easier to review and to revert.
- Describe what you changed and why, and how you verified it. For detection changes, say what footage or scenario you used.
- Add a line under `Unreleased` in `CHANGELOG.md` for any user-visible change.

## Reporting bugs

Please include the `safework-cv` version, operating system, Python version, the source type (file, RTSP, webcam), your configuration file and the log output with `--log-level DEBUG`. For detection problems, the most useful report is a short clip together with the time at which the problem occurs.

Never attach footage that shows identifiable people unless you have their permission. Blur faces or describe the scene instead.

## Security issues

Do not open a public issue for vulnerabilities. Follow [SECURITY.md](SECURITY.md).

## Licence

By contributing you agree that your contribution is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE), the same licence as the project.
