"""
Command-line interface for training and evaluating the USV RL agent.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    try:
        from tensorboardX import SummaryWriter  # type: ignore
    except ModuleNotFoundError:

        class SummaryWriter:  # type: ignore
            """Graceful fallback when TensorBoard is unavailable."""

            def __init__(self, *_, **__):
                print("TensorBoard not available; metrics will not be logged.")

            def add_scalar(self, *_, **__):
                return

            def close(self):
                return
from tqdm import tqdm

from marine_environment_simulation import MarineEnvironmentSimulation
from usv_obstacles import ObstacleWorld

from .buffer import ReplayBufferSeq
from .models import ActorLSTM, CriticWithAuxLSTM
from .rollout import VectorizedCollector
from .trainer import Trainer, TrainerConfig
from .utils import build_obs_layout, ensure_dir, load_yaml_config, max_batch_finder, set_seed
from . import viz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("USV RL")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--save-dir", default="outputs")
    parser.add_argument("--tb-dir", default="runs")
    parser.add_argument("--map-dpi", type=int, default=220)
    parser.add_argument("--make-map-only", action="store_true")
    parser.add_argument("--train", dest="train", action="store_true", default=True)
    parser.add_argument("--eval", dest="train", action="store_false")
    parser.add_argument("--n-env", type=int, default=64)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=16)
    parser.add_argument("--auto-batch", action="store_true")
    parser.add_argument("--gpu-replay", action="store_true")
    parser.add_argument("--topN", type=int, default=16)
    parser.add_argument("--obs-include-wind", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--no-aux", action="store_true")
    parser.add_argument("--no-attn", action="store_true")
    parser.add_argument("--ff-only", action="store_true")
    parser.add_argument("--mask-occlusion", action="store_true")
    parser.add_argument("--occlude-t0", type=int, default=200)
    parser.add_argument("--occlude-len", type=int, default=80)
    parser.add_argument("--occlude-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--attn-viz-every", type=int, default=5000)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--checkpoint", default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def build_models(cfg: dict, obs_dim: int, action_limits: torch.Tensor, device: torch.device, args: argparse.Namespace):
    actor = ActorLSTM(
        obs_dim=obs_dim,
        action_limits=action_limits,
        top_n=args.topN,
        include_attention=not args.no_attn,
        ff_only=args.ff_only,
    ).to(device)
    critic = CriticWithAuxLSTM(
        obs_dim=obs_dim,
        action_dim=action_limits.numel(),
        top_n=args.topN,
        include_attention=not args.no_attn,
        ff_only=args.ff_only,
    ).to(device)
    target_critic = CriticWithAuxLSTM(
        obs_dim=obs_dim,
        action_dim=action_limits.numel(),
        top_n=args.topN,
        include_attention=not args.no_attn,
        ff_only=args.ff_only,
    ).to(device)
    target_critic.load_state_dict(critic.state_dict())
    if args.compile:
        actor = torch.compile(actor)
        critic = torch.compile(critic)
    return actor, critic, target_critic


def maybe_load_checkpoint(path: str | None, actor, critic, target_critic, device: torch.device) -> int:
    if not path:
        return 0
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")
    payload = torch.load(ckpt_path, map_location=device)
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    target_critic.load_state_dict(payload["target"])
    return int(payload.get("step", 0))


def warmup_buffer(collector: VectorizedCollector, actor, buffer: ReplayBufferSeq, normalizers: dict, n_env: int, iters: int):
    actor.eval()
    for _ in range(iters):
        collector.collect(actor, buffer, normalizers, n_env)
    actor.train()


def run_evaluation(actor, collector: VectorizedCollector, args: argparse.Namespace, save_dir: Path, occlusion: bool = False) -> None:
    actor.eval()
    occ = None
    if occlusion:
        occ = {"t0": args.occlude_t0, "len": args.occlude_len, "prob": args.occlude_prob}
    metrics = collector.collect(actor, buffer=None, normalizers={}, batch_envs=args.n_env, store=False, occlusion=occ)
    if not metrics:
        print("No evaluation trajectories collected.")
        return
    summary = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}
    csv_path = save_dir / ("occlusion_metrics.csv" if occlusion else "eval_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("metric,value\n")
        for key, val in summary.items():
            handle.write(f"{key},{val}\n")
    print(f"Saved evaluation metrics to {csv_path}")


def train_loop(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg = load_yaml_config(args.config)
    cfg.update(
        {
            "seq_len": args.seq,
            "burn_in": args.burn_in,
            "topN": args.topN,
            "include_wind": args.obs_include_wind,
            "gamma": args.gamma,
            "tau": args.tau,
            "batch": args.batch,
        }
    )

    save_dir = ensure_dir(args.save_dir)
    tb_dir = ensure_dir(args.tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    sim = MarineEnvironmentSimulation(map_extent=cfg["L"])
    world = ObstacleWorld(L=cfg["L"])
    map_path = save_dir / "map_snapshot.png"
    viz.render_map_snapshot(map_path, sim, world, cfg["L"], dpi=args.map_dpi)
    if args.make_map_only:
        print(f"Saved map snapshot to {map_path}")
        return

    obs_slices, obs_dim = build_obs_layout(args.topN, args.obs_include_wind)
    act_limits = torch.tensor([cfg["U_max"], cfg["U_max"], 0.3], device=device)

    actor, critic, target_critic = build_models(cfg, obs_dim, act_limits, device, args)
    start_step = maybe_load_checkpoint(args.checkpoint, actor, critic, target_critic, device)
    buffer = ReplayBufferSeq(cfg["replay_capacity"], cfg["seq_len"], device, gpu_storage=args.gpu_replay)
    collector = VectorizedCollector(cfg, args.topN, args.obs_include_wind, act_limits.cpu().numpy(), device)
    normalizers = {}

    if not args.train:
        run_evaluation(actor, collector, args, save_dir, occlusion=args.mask_occlusion)
        return

    warmup_iters = max(1, args.batch // max(1, args.n_env))
    warmup_buffer(collector, actor, buffer, normalizers, args.n_env, warmup_iters)

    trainer_cfg = TrainerConfig(
        batch_size=args.batch,
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        actor_lr=cfg["actor_lr"],
        critic_lr=cfg["critic_lr"],
        alpha_lr=cfg["alpha_lr"],
        target_entropy=cfg.get("target_entropy", -3.0),
        burn_in=cfg["burn_in"],
        use_amp=args.amp,
        u_max=cfg["U_max"],
        aux_weight=0.0 if args.no_aux else cfg.get("aux_weight", 0.2),
    )
    trainer = Trainer(actor, critic, target_critic, buffer, trainer_cfg, device, writer)

    if args.auto_batch:
        snapshot = {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "target": target_critic.state_dict(),
            "alpha": trainer.log_alpha.detach().clone(),
            "actor_opt": trainer.actor_opt.state_dict(),
            "critic_opt": trainer.critic_opt.state_dict(),
            "alpha_opt": trainer.alpha_opt.state_dict(),
            "scaler": trainer.scaler.state_dict(),
        }

        def _trial(bs: int):
            trainer.cfg.batch_size = bs
            orig_writer = trainer.writer
            trainer.writer = None
            trainer.train_step(-1)
            trainer.writer = orig_writer
            actor.load_state_dict(snapshot["actor"])
            critic.load_state_dict(snapshot["critic"])
            target_critic.load_state_dict(snapshot["target"])
            trainer.log_alpha.data.copy_(snapshot["alpha"])
            trainer.actor_opt.load_state_dict(snapshot["actor_opt"])
            trainer.critic_opt.load_state_dict(snapshot["critic_opt"])
            trainer.alpha_opt.load_state_dict(snapshot["alpha_opt"])
            trainer.scaler.load_state_dict(snapshot["scaler"])

        best = max_batch_finder(_trial, args.batch)
        trainer.cfg.batch_size = max(1, int(best * 0.9))

    global_step = start_step
    progress = tqdm(range(args.steps), desc="Train", ncols=120)
    for step in progress:
        collector.collect(actor, buffer, normalizers, args.n_env)
        try:
            metrics = trainer.train_step(global_step)
        except RuntimeError as exc:
            if "out of memory" in str(exc):
                torch.cuda.empty_cache()
                trainer.cfg.batch_size = max(1, int(trainer.cfg.batch_size * 0.8))
                print(
                    f"OOM detected during training; reducing batch size to {trainer.cfg.batch_size} and retrying."
                )
                continue
            raise
        if torch.cuda.is_available():
            metrics["vram_gb"] = float(torch.cuda.memory_allocated(device) / (1024**3))
        log_line = f"[{global_step:07d}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(log_line)
        if args.log_interval and global_step % args.log_interval == 0:
            for key, val in metrics.items():
                writer.add_scalar(f"train/{key}", val, global_step)
        if args.save_every and global_step % args.save_every == 0 and global_step > 0:
            path = save_dir / f"checkpoint_{global_step:07d}.pt"
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "target": target_critic.state_dict(),
                    "cfg": cfg,
                    "step": global_step,
                },
                path,
            )
        if args.attn_viz_every and global_step % args.attn_viz_every == 0:
            try:
                attn_batch = buffer.sample(1, device)
                with torch.no_grad():
                    _, _, _, weights = actor(
                        attn_batch["obs"],
                        attn_batch["obs_mask"],
                        attn_batch["attn_mask"],
                        cfg["burn_in"],
                        deterministic=True,
                    )
                if weights is not None:
                    attn_np = weights.mean(dim=1)[0].detach().cpu().numpy()
                    traj_full = attn_batch["meta"][0].detach().cpu().numpy()[:, :2]
                    mask_np = attn_batch["obs_mask"][0].detach().cpu().numpy() > 0.5
                    traj = traj_full[mask_np]
                    if traj.size > 0:
                        viz.draw_attention_overlay(traj, attn_np, collector.world, save_dir / f"attn_{global_step:07d}.png")
            except Exception as exc:
                print(f"Attention visualization skipped: {exc}")
        global_step += 1
    writer.close()
    print("Training complete.")
    if args.mask_occlusion:
        run_evaluation(actor, collector, args, save_dir, occlusion=True)


def main():
    args = parse_args()
    train_loop(args)


if __name__ == "__main__":
    main()
