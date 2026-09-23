#!/usr/bin/env python3
"""Train the MP11 extra-credit deep-Q pong player.

The learner itself lives in ``submitted.py`` (that is the only file the
autograder receives); this script is the harness around it: it drives the
pong environment, logs to Weights & Biases in **offline** mode, evaluates a
greedy copy of the policy on the way, and keeps the best checkpoint.

Examples
--------
    # single run, the configuration that produced trained_model.pkl
    python train_deepq.py --arch resnet --frames 600000

    # compare all four backbones, each in its own wandb run
    python train_deepq.py --arch mlp cnn resnet transformer --frames 400000

Offline wandb runs land in ``wandb/`` and can be pushed later with
``wandb sync wandb/offline-run-*``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

import pong
import submitted
from submitted import DeepQConfig, deep_q

# wandb must be told to stay offline before it is imported, otherwise it may
# try to reach the network while resolving the default entity.
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")

RUNS_DIR = Path(__file__).resolve().parent / "runs"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


class NullRun:
    """Stand-in for a wandb run so the script still works without wandb."""

    name = "wandb-disabled"
    dir = str(RUNS_DIR)

    def log(self, *_args, **_kwargs) -> None:
        pass

    def summary_update(self, *_args, **_kwargs) -> None:
        pass

    def finish(self) -> None:
        pass


def start_run(project: str, name: str, config: Dict, enabled: bool = True):
    """Open an offline wandb run, or a no-op stub if wandb is unavailable."""
    if not enabled:
        return NullRun(), False
    try:
        import wandb
    except ImportError:
        print("[warn] wandb is not installed -- logging to JSON only")
        return NullRun(), False

    run = wandb.init(
        project=project,
        name=name,
        mode="offline",
        dir=str(RUNS_DIR),
        config=config,
    )
    return run, True


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #


def make_game(agent, ball_speed: float, paddle_speed: float) -> pong.PongGame:
    """A headless, unquantized pong game bound to ``agent``."""
    return pong.PongGame(
        ball_speed=ball_speed,
        paddle_speed=paddle_speed,
        learner=agent,
        visible=False,
        state_quantization=None,
    )


def play_games(
    agent: deep_q,
    n_games: int,
    ball_speed: float = 4,
    paddle_speed: float = 8,
    max_frames_per_game: int = 50_000,
    record_trajectory: bool = False,
) -> Dict[str, object]:
    """Play ``n_games`` greedy games and report the per-game scores.

    This mirrors ``pong.PongGame.run`` exactly -- the score credited to a game
    is ``game.score`` as it stood at the start of the frame on which the ball
    was missed -- but adds a frame cap so that an agent which never loses
    cannot hang the evaluation.

    @return: dict with "scores" (list of int) and, optionally, "trajectory".
    """
    was_training = agent.training
    agent.eval_mode()

    game = make_game(agent, ball_speed, paddle_speed)
    state = game.state_init()
    scores: List[int] = []
    trajectory: List[tuple] = []
    frames_this_game = 0

    while len(scores) < n_games:
        action = agent.act(state)
        score_before = game.score
        if record_trajectory:
            trajectory.append((state[0], state[1], state[4], game.score, action))

        newstate, reward = game.update(state, action * paddle_speed)
        agent.learn(state, action, reward, newstate)
        frames_this_game += 1

        if reward < 0:
            scores.append(score_before)
            frames_this_game = 0
        elif frames_this_game >= max_frames_per_game:
            scores.append(game.score)
            game.score = 0
            newstate = game.state_init()
            agent.end_episode()
            frames_this_game = 0

        state = newstate

    if was_training:
        agent.train_mode()

    result: Dict[str, object] = {"scores": scores}
    if record_trajectory:
        result["trajectory"] = trajectory
    return result


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train(arch: str, args: argparse.Namespace) -> Dict[str, object]:
    """Train one backbone and return its history. Saves the best checkpoint."""
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    config = DeepQConfig(
        arch=arch,
        n_frames=args.frames_stacked,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        warmup=args.warmup,
        train_every=args.train_every,
        target_sync=args.target_sync,
        n_step=args.n_step,
        shaping=args.shaping,
        eps_start=args.eps_start,
        eps_decay_steps=args.eps_decay or max(1, args.frames // 4),
        device=args.device,
        seed=seed,
    )
    agent = deep_q(args.alpha, args.epsilon, args.gamma, args.nfirst, config=config)

    run_name = args.run_name or f"{arch}-f{args.frames_stacked}-s{seed}"
    run, live = start_run(
        args.project,
        run_name,
        {**asdict(config), "frames": args.frames, "seed": seed, "parameters": agent.n_parameters},
        enabled=not args.no_wandb,
    )
    print(
        f"\n=== {arch}: {agent.n_parameters:,} parameters on {agent.device} "
        f"| {args.frames:,} frames | wandb {'offline' if live else 'disabled'} ==="
    )

    game = make_game(agent, args.ball_speed, args.paddle_speed)
    state = game.state_init()

    history: Dict[str, List] = {
        "game_frame": [], "game_score": [],
        "eval_frame": [], "eval_mean": [], "eval_max": [],
        "loss_frame": [], "loss": [], "epsilon": [], "q_mean": [],
    }
    best_eval = -np.inf
    best_path = RUNS_DIR / run_name / "best.pkl"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    frames_this_game = 0
    recent: List[int] = []
    started = time.time()

    for frame in range(1, args.frames + 1):
        action = agent.act(state)
        score_before = game.score
        newstate, reward = game.update(state, action * args.paddle_speed)
        agent.learn(state, action, reward, newstate)
        frames_this_game += 1

        if reward < 0:
            recent.append(score_before)
            history["game_frame"].append(frame)
            history["game_score"].append(score_before)
            frames_this_game = 0
        elif frames_this_game >= args.max_episode_frames:
            # Truncate an endless rally so one game cannot monopolise training.
            recent.append(game.score)
            history["game_frame"].append(frame)
            history["game_score"].append(game.score)
            game.score = 0
            newstate = game.state_init()
            agent.end_episode()
            frames_this_game = 0

        state = newstate

        if frame % args.log_every == 0:
            window = recent[-100:]
            metrics = {
                "frame": frame,
                "epsilon": agent.epsilon_now,
                "buffer": len(agent.buffer),
                "updates": agent.updates,
                "score/mean_100": float(np.mean(window)) if window else 0.0,
                "score/max_100": float(np.max(window)) if window else 0.0,
                "games": len(recent),
                "fps": frame / max(1e-9, time.time() - started),
                **{f"train/{k}": v for k, v in agent.metrics.items()},
            }
            run.log(metrics, step=frame)
            history["loss_frame"].append(frame)
            history["loss"].append(agent.metrics.get("loss", float("nan")))
            history["q_mean"].append(agent.metrics.get("q_mean", float("nan")))
            history["epsilon"].append(agent.epsilon_now)

        if frame % args.eval_every == 0:
            scores = play_games(agent, args.eval_games, args.ball_speed, args.paddle_speed,
                                max_frames_per_game=args.eval_max_frames)["scores"]
            mean, peak = float(np.mean(scores)), int(np.max(scores))
            history["eval_frame"].append(frame)
            history["eval_mean"].append(mean)
            history["eval_max"].append(peak)
            run.log({"eval/mean": mean, "eval/max": peak}, step=frame)
            # Snapshot every evaluation. Late-training collapse is severe here
            # (last.pkl routinely scores 10-20x worse than best.pkl), and once
            # the agent outgrows the eval cap "best" is picked from saturated
            # numbers -- so keep the candidates and score them properly later.
            agent.save(best_path.parent / f"ckpt_{frame:08d}.pkl")
            flag = ""
            if mean > best_eval:
                best_eval, flag = mean, "  <- best"
                agent.save(best_path)
            print(
                f"  frame {frame:>8,}  eps {agent.epsilon_now:5.3f}  "
                f"train100 {np.mean(recent[-100:]) if recent else 0:6.2f}  "
                f"eval {mean:6.2f} (max {peak}){flag}"
            )
            # play_games leaves the frame stack mid-episode; restart cleanly.
            agent.end_episode()
            state = game.state_init()
            game.score = 0
            frames_this_game = 0

    last_path = RUNS_DIR / run_name / "last.pkl"
    agent.save(last_path)

    # Final evaluation uses the best checkpoint, exactly as the autograder will.
    # Evaluate last.pkl separately with evaluate.py: once the agent outgrows the
    # periodic eval's frame cap, "best" is chosen from saturated numbers and the
    # final weights are often the better model.
    if best_path.exists():
        agent.load(best_path)
    final = play_games(agent, args.final_games, args.ball_speed, args.paddle_speed,
                       max_frames_per_game=args.final_max_frames)["scores"]
    final_mean = float(np.mean(final))
    print(f"  final greedy eval over {args.final_games} games: mean {final_mean:.2f}")

    run.log({"eval/final_mean": final_mean}, step=args.frames)
    if live:
        run.summary["eval/final_mean"] = final_mean
        run.summary["eval/best_mean"] = best_eval
        run.summary["parameters"] = agent.n_parameters
    run.finish()

    history.update(
        arch=arch,
        run_name=run_name,
        parameters=agent.n_parameters,
        config=asdict(agent.config),
        final_scores=final,
        final_mean=final_mean,
        best_eval=float(best_eval),
        wall_seconds=time.time() - started,
    )
    out = RUNS_DIR / run_name / "history.json"
    out.write_text(json.dumps(history, indent=2))
    print(f"  history -> {out}")
    return history


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--arch", nargs="+", default=["resnet"],
                   choices=sorted(submitted.BACKBONES), help="backbone(s) to train")
    p.add_argument("--frames", type=int, default=600_000, help="environment frames per run")
    p.add_argument("--frames-stacked", type=int, default=4, help="states per observation")

    p.add_argument("--alpha", type=float, default=0.05, help="MP API learning rate (stored only)")
    p.add_argument("--epsilon", type=float, default=0.02, help="floor of the epsilon schedule")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--nfirst", type=int, default=5, help="MP API argument (stored only)")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--warmup", type=int, default=5_000)
    p.add_argument("--train-every", type=int, default=4)
    p.add_argument("--target-sync", type=int, default=1_000)
    p.add_argument("--n-step", type=int, default=1, help="multi-step return length")
    p.add_argument("--shaping", type=float, default=0.0,
                   help="potential-based shaping weight (0 disables)")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-decay", type=int, default=0, help="0 means frames // 4")

    p.add_argument("--ball-speed", type=float, default=4)
    p.add_argument("--paddle-speed", type=float, default=8)
    p.add_argument("--max-episode-frames", type=int, default=5_000)

    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--eval-max-frames", type=int, default=6_000,
                   help="frame cap per evaluation game; keeps mid-training evals cheap")
    p.add_argument("--final-games", type=int, default=20)
    p.add_argument("--final-max-frames", type=int, default=30_000)
    p.add_argument("--log-every", type=int, default=1_000)

    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=4,
                   help="torch CPU threads; keep it small when running several archs at once")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", default="mp11-deepq-pong")
    p.add_argument("--run-name", default=None, help="only valid with a single --arch")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--promote", default=None,
                   help="copy the winning checkpoint here, e.g. trained_model.pkl")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.run_name and len(args.arch) > 1:
        raise SystemExit("--run-name can only be used with a single --arch")
    # The networks are tiny and the environment loop is serial, so letting torch
    # grab every core only buys contention -- especially with runs in parallel.
    torch.set_num_threads(max(1, args.threads))
    RUNS_DIR.mkdir(exist_ok=True)

    histories = [train(arch, args) for arch in args.arch]

    print("\n=== summary ===")
    print(f"{'arch':<13}{'params':>10}{'best eval':>11}{'final':>9}{'minutes':>9}")
    for h in sorted(histories, key=lambda h: -h["final_mean"]):
        print(f"{h['arch']:<13}{h['parameters']:>10,}{h['best_eval']:>11.2f}"
              f"{h['final_mean']:>9.2f}{h['wall_seconds'] / 60:>9.1f}")

    # Named after the archs in *this* invocation: training one arch per process
    # (the usual way to use the machine) would otherwise have every run clobber
    # a single shared summary.json. The per-run history.json files stay the
    # authoritative record either way.
    summary = RUNS_DIR / ("summary_" + "_".join(h["arch"] for h in histories) + ".json")
    summary.write_text(json.dumps(
        [{k: h[k] for k in ("arch", "run_name", "parameters", "best_eval", "final_mean",
                            "final_scores", "wall_seconds")} for h in histories], indent=2))
    print(f"summary -> {summary}")

    if args.promote:
        import shutil
        winner = max(histories, key=lambda h: h["final_mean"])
        src = RUNS_DIR / winner["run_name"] / "best.pkl"
        shutil.copyfile(src, args.promote)
        print(f"promoted {winner['arch']} ({winner['final_mean']:.2f}) -> {args.promote}")


if __name__ == "__main__":
    main()
