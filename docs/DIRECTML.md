# DirectML and GigaLearnRL

**`pip install torch-directml` does not accelerate GigaLearnBot.**

PPO is C++ LibTorch. DirectML is a Python package (`PrivateUse1`). No supported LibTorch+DirectML build for this trainer ([DirectML #247](https://github.com/microsoft/DirectML/issues/247)).

| Stack | Speeds GigaLearnBot? |
|---|---|
| Windows HIP + HIP RocketSimCuda | Env yes / PPO = CPU (threaded) |
| torch-directml (Python) | No |
| AutoTrainer deps | Optional; AT does not need DirectML |
| WSL2 + ROCm LibTorch | Optional advanced ([`AMD_WSL2.md`](AMD_WSL2.md)) |

`GIGA_TORCH_DEVICE=directml` / `privateuseone`: warn and use CPU. Prefer `cpu`, `cuda`, `gpu`, or `auto`.

For RX 6600 XT on Windows: [`AMD.md`](AMD.md) -> HIP build, arenas 2048, tune `GIGA_TORCH_THREADS`. IT: [`../INIZIO_RAPIDO_IT.md`](../INIZIO_RAPIDO_IT.md).
