# scripts

Use `setup_env.sh` to reproduce a locked environment and apply the required integration patches. `build_integration.py` validates exact upstream versions, revisions, and source hashes before making any environment-local changes.

Training and evaluation launchers use repository-relative configuration paths and expect the matching environment profile to have been built first.
