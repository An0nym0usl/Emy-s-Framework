# GigaLearnRL Python SDK (`giga-sdk`)

A small, dependency-free Python wrapper over the framework's **checkpoint box** (the
`checkpoints/` folder). It mirrors the C++ `GGL::CheckpointManager` on-disk format, so the
Python and C++ SDKs always agree.

## Install

```bash
pip install -e sdk/python
```

This exposes the `giga` command-line tool and the importable `giga_sdk` package.

## Command line

```bash
giga --checkpoints build/Release/checkpoints list      # list every checkpoint + flags
giga info 47694027776                                  # full metadata for one checkpoint
giga info                                              # metadata for the latest checkpoint
giga validate                                          # validate the latest checkpoint
giga export 47694027776 ./deploy_model                 # copy inference files (POLICY/SHARED_HEAD)
giga prune --keep 8                                    # delete old checkpoints
giga versions                                          # list policy-version snapshots
giga deploy                                            # run the bot from the latest valid checkpoint
giga deploy --checkpoint 47694027776 --exe build/Release/GigaLearnBot.exe
```

The checkpoint folder defaults to `$GIGA_CHECKPOINT_DIR` or `./checkpoints`.

## Library

```python
from giga_sdk import CheckpointManager

mgr = CheckpointManager("build/Release/checkpoints")
for c in mgr.list():
    print(c.timesteps, c.is_valid_for_inference, c.total_iterations)

latest = mgr.latest_valid_for_inference()
ok, err = mgr.validate(latest.path)
mgr.export(latest.timesteps, "deploy_model")
```

## Optional native binding

`sdk/python/native/` contains an optional pybind11 module (`giga_sdk_native`) that wraps the
C++ `GGL::CheckpointManager` directly. It is **not required** — the pure-Python implementation is
the default. Build it only for native acceleration:

```bash
cmake -S sdk/python/native -B build_sdk
cmake --build build_sdk --config Release
```

`giga deploy` / `giga info` connect the SDK to the C++ core: checkpoint management is handled in
Python, while inference and RLBot deployment run the compiled `GigaLearnBot` executable on the
selected checkpoint.
