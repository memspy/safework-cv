# Security policy

## Supported versions

SafeWork CV is in alpha. Security fixes are applied to the latest release on the `main` branch.

## Reporting a vulnerability

Please do not report vulnerabilities in public issues or pull requests.

Use GitHub private vulnerability reporting instead: open the **Security** tab of the repository and choose **Report a vulnerability**. Describe the problem, the affected version and how to reproduce it. The maintainers aim to acknowledge a report within a week and to agree on a disclosure date with you once the issue is understood.

## Operating notes

These are properties of the design that matter when deploying SafeWork CV, not vulnerabilities in themselves.

- **Model weights are code.** PyTorch `.pt` files are pickles and can execute arbitrary code when loaded. Use `--weights`, `--hazard-weights`, `yolo_weights` and `hazard_weights` only with files from sources you trust.
- **Webhooks are not authenticated.** Events are sent as plain JSON without credentials. Use an `https://` endpoint and restrict access to it at the network level or through a gateway you control.
- **Stream credentials.** Credentials in an RTSP URL are masked in logs but are visible in the process list and in shell history when passed on the command line. Use a read-only camera account and restrict who can see the process list.
- **Footage is personal data.** Snapshots and annotated videos show people. Store them on encrypted storage, limit access and set a retention period.
- **Container.** The Docker image runs as an unprivileged user.
