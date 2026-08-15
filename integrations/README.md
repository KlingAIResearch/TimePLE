# Integration patches

TimePLE integrations are built on top of exact upstream releases during environment setup. This repository stores:

- a manifest containing the required distribution version, source revision where applicable, and before/after hashes;
- a unified patch containing only the TimePLE integration delta.

Complete upstream source trees are not vendored. Run `bash scripts/setup_env.sh <profile>` to install and validate the matching environment. Every modified upstream file receives a visible TimePLE modification notice when the patch is applied.
