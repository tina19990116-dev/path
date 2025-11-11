# USV RL

GPU-ready LSTM-SAC baseline for unmanned surface vehicle path planning with attention over obstacle tokens and auxiliary physics prediction.

## Quick Start
```
python -m usv_rl.main --make-map-only --save-dir outputs
python -m usv_rl.main --device cuda --amp --auto-batch --gpu-replay --save-dir outputs --tb-dir runs --steps 200000
python -m usv_rl.main --device cuda --mask-occlusion --occlude-t0 200 --occlude-len 80 --occlude-prob 0.5
python -m usv_rl.ablation --device cuda --save-dir outputs/ablations
```

## Components
- **Normalization**: Running mean/variance on observations (see `usv_rl/utils.py`).
- **Reward decomposition**: Logged per-term scalars (`reward.py`) ensuring shape, collision, energy, smoothness, time, and goal bonuses.
- **Attention visualization**: Generated via `viz.draw_attention_overlay` using stored multi-head weights.

## Outputs
- Training curves and TensorBoard logs land under `--save-dir`/`--tb-dir`.
- Occlusion and ablation summaries are exported as CSV/JSON for downstream plotting.
- High-DPI map snapshots stored at `outputs/map_snapshot.png` by default.
