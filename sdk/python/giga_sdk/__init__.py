"""GigaLearnRL Python SDK.

A thin, dependency-free wrapper over the framework's "checkpoint box" (the ``checkpoints/``
folder that training writes to and deployment reads from). It mirrors the C++
``GGL::CheckpointManager`` on-disk format exactly, so the two stay in sync.

Typical use::

    from giga_sdk import CheckpointManager
    mgr = CheckpointManager("build/Release/checkpoints")
    for ckpt in mgr.list():
        print(ckpt.timesteps, ckpt.is_valid_for_inference)
    latest = mgr.latest_valid_for_inference()

Or from the command line::

    giga list
    giga info <timesteps>
    giga export <timesteps> ./deploy_model
    giga deploy --checkpoint <timesteps>
"""

from .checkpoints import CheckpointInfo, CheckpointManager, default_checkpoint_dir

__all__ = ["CheckpointInfo", "CheckpointManager", "default_checkpoint_dir"]
__version__ = "1.0.0"
