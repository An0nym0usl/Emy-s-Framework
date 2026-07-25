# Opponent pool (`opponents/`)

External sparring bots for mid-run OpponentPool / league mix. Paths in the manifest are **relative** to this folder.

## Default

`opponents.json` ships with **`"entries": []`** — self-play only until you add bots. No model folders are shipped.

## How to add an opponent

1. Create e.g. `opponents/mio_bot/` and put weights there.
2. Append an entry to `opponents.json`:

```json
{
  "entries": [
    {
      "name": "nexto",
      "type": "nexto_jit",
      "model": "nexto/nexto-model.pt",
      "weight": 0.85,
      "beta": 1.0
    },
    {
      "name": "necto",
      "type": "necto_jit",
      "model": "necto/necto-model.pt",
      "weight": 0.70
    }
  ]
}
```

3. Types: `nexto_jit`, `necto_jit`, or `gigalearn` (same-arch ckpt with `POLICY.lt` + `SHARED_HEAD.lt`).
4. Copy the same tree under `build/Release/opponents/` if the exe runs from Release (cwd = exe dir).
5. Restart train / AutoTrainer. Missing files are skipped without crashing.

## Manifest fields

| Field | Meaning |
|---|---|
| `name` | Metric label |
| `type` | `nexto_jit` / `necto_jit` / `gigalearn` |
| `model` | Relative path to `.pt` or ckpt folder |
| `weight` | Relative pick probability |
| `beta` | Nexto only (`1` = greedy) |
| `beat_bonus_scale` / `concede_penalty_scale` | Pool beat/concede scales |

## Where to get community models (optional)

- **Nexto**: RLBot Pack → place as `opponents/nexto/nexto-model.pt`
- **Necto**: [Rolv-Arild/Necto](https://github.com/Rolv-Arild/Necto) → `opponents/necto/necto-model.pt`

More: `docs/CUSTOMIZE.md`.
