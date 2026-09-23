#!/usr/bin/env python3
"""Measure how well a trained deep-Q pong player really plays.

``train_deepq.py`` caps its evaluation games so that training stays cheap, and
a strong learner spends most of its games sitting on that cap -- the reported
average then says more about the cap than about the policy. This script plays
uncapped-in-spirit games (a very large frame budget) and runs them in parallel
worker processes, which is what makes a four-figure score measurable at all.

    python evaluate.py --model trained_model_transformer.pkl \
        --games 10 --max-frames 400000 --workers 10

Every game is reported with the frame budget it was given and whether it hit
it, because a game that ends on the budget is a lower bound, not a score.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent


def play_one(job: Dict) -> Dict:
    """Play a single greedy game and report how it ended."""
    import torch

    torch.set_num_threads(job["threads"])
    import pong
    import submitted

    random.seed(job["seed"])
    np.random.seed(job["seed"])

    agent = submitted.deep_q(0.05, 0.05, 0.99, 5, device="cpu")
    agent.load(job["model"])
    agent.eval_mode()

    game = pong.PongGame(ball_speed=job["ball_speed"], paddle_speed=job["paddle_speed"],
                         learner=agent, visible=False, state_quantization=None)
    state = game.state_init()

    started = time.time()
    for frame in range(1, job["max_frames"] + 1):
        action = agent.act(state)
        score_before = game.score
        newstate, reward = game.update(state, action * job["paddle_speed"])
        agent.learn(state, action, reward, newstate)   # eval mode: bookkeeping only
        if reward < 0:
            return {"seed": job["seed"], "score": score_before, "frames": frame,
                    "capped": False, "seconds": time.time() - started}
        state = newstate

    return {"seed": job["seed"], "score": game.score, "frames": job["max_frames"],
            "capped": True, "seconds": time.time() - started}


def summarise(results: List[Dict], budget: int) -> Dict:
    scores = np.array([r["score"] for r in results], dtype=float)
    capped = sum(r["capped"] for r in results)
    frames = np.array([r["frames"] for r in results], dtype=float)
    per_hit = frames.sum() / max(1.0, scores.sum())
    return {
        "games": len(results),
        "frame_budget": budget,
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "min": int(scores.min()),
        "max": int(scores.max()),
        "capped_games": int(capped),
        "frames_per_hit": float(per_hit),
        "scores": sorted((int(s) for s in scores), reverse=True),
    }


def report(name: str, stats: Dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  games            {stats['games']}  (frame budget {stats['frame_budget']:,} each)")
    print(f"  mean score       {stats['mean']:.2f}")
    print(f"  median score     {stats['median']:.1f}")
    print(f"  min / max        {stats['min']} / {stats['max']}")
    print(f"  hit the budget   {stats['capped_games']} / {stats['games']}"
          + ("   <- mean is a LOWER BOUND" if stats["capped_games"] else ""))
    print(f"  frames per hit   {stats['frames_per_hit']:.1f}")
    print(f"  scores           {stats['scores']}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="trained_model.pkl")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=400_000,
                   help="frame budget per game; a game that reaches it is a lower bound")
    p.add_argument("--workers", type=int, default=0, help="0 means one per game")
    p.add_argument("--threads", type=int, default=4, help="torch threads per worker")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ball-speed", type=float, default=4)
    p.add_argument("--paddle-speed", type=float, default=8)
    p.add_argument("--out", default=None, help="write the summary to this JSON file")
    args = p.parse_args(argv)

    jobs = [{"model": args.model, "seed": args.seed + i, "max_frames": args.max_frames,
             "threads": args.threads, "ball_speed": args.ball_speed,
             "paddle_speed": args.paddle_speed} for i in range(args.games)]

    workers = args.workers or args.games
    started = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(play_one, jobs)

    stats = summarise(results, args.max_frames)
    stats["model"] = args.model
    stats["wall_seconds"] = time.time() - started
    report(Path(args.model).name, stats)
    print(f"  wall clock       {stats['wall_seconds'] / 60:.1f} min")

    if args.out:
        Path(args.out).write_text(json.dumps(stats, indent=2))
        print(f"  summary -> {args.out}")


if __name__ == "__main__":
    main()
