#!/usr/bin/env python3
"""Figures for the MP11 extra-credit deep-Q pong player.

Subcommands
-----------
    curves    training and evaluation curves for one run
    compare   small multiples + final scores across backbones
    policy    what the trained network believes: Q landscape and policy maps
    rollout   an animated greedy episode (GIF) with a live Q-value panel
    all       every figure above

    python visualize.py all --model trained_model.pkl

Figures are written to ``figures/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import viz_style as vs

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
FIG_DIR = HERE / "figures"

# Environment geometry, mirrored from pong.PongGame's defaults.
GAME_W, GAME_H = 600.0, 400.0
BALL_R, PADDLE_W, PADDLE_H = 20.0, 8.0, 80.0


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def rolling(values: Sequence[float], window: int) -> np.ndarray:
    """Centred-right rolling mean, same length as the input."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    window = max(1, min(window, arr.size))
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, arr[0]), arr])
    return np.convolve(padded, kernel, mode="valid")


def load_history(run_dir: Path) -> Dict:
    return json.loads((run_dir / "history.json").read_text())


def find_runs(explicit: Optional[List[str]]) -> List[Path]:
    if explicit:
        return [Path(p) if Path(p).is_dir() else RUNS_DIR / p for p in explicit]
    return sorted(p.parent for p in RUNS_DIR.glob("*/history.json"))


def thousands(x, _pos) -> str:
    return f"{x / 1000:g}k" if x >= 1000 else f"{x:g}"


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(HERE)}")
    return path


def dot_row(ax, y: float, value: float, color: str, left: float) -> None:
    """A dot plot row: a recessive guide rule plus a dot at the value.

    Used instead of a bar whenever the axis is logarithmic. On a log scale a
    bar's *length* no longer encodes its magnitude, so the dot carries the
    value by position and the rule is only there to lead the eye.
    """
    ax.plot([left, value], [y, y], color=vs.GRID, linewidth=1.0, zorder=2)
    ax.plot([value], [y], marker="o", markersize=10, color=color,
            markeredgecolor=vs.SURFACE, markeredgewidth=1.5, zorder=4)


def rounded_bar(ax, y: float, value: float, color: str, thickness: float = 9.0) -> None:
    """A thin horizontal bar with a rounded data-end, flat against x=0.

    The round cap at the baseline is clipped away by the axes, which leaves
    exactly the anatomy the style guide asks for.
    """
    ax.plot([0, value], [y, y], color=color, linewidth=thickness,
            solid_capstyle="round", clip_on=True, zorder=3)


# --------------------------------------------------------------------------- #
# curves
# --------------------------------------------------------------------------- #


def figure_curves(history: Dict, name: str = "fig_learning.png") -> Path:
    arch = history["arch"]
    color = vs.SERIES_BY_NAME.get(arch, vs.SERIES[0])
    # The evaluation curve must never land on the backbone's own colour --
    # for the cnn run the two would otherwise both be orange.
    eval_color = vs.SERIES[1] if color != vs.SERIES[1] else vs.SERIES[0]

    fig = plt.figure(figsize=(11, 7.6))
    grid = GridSpec(2, 3, figure=fig, height_ratios=[2.25, 1],
                    hspace=0.42, wspace=0.26, top=0.855, bottom=0.08, left=0.065, right=0.975)

    # -- main panel: score per game, its rolling mean, and the greedy evals --
    ax = fig.add_subplot(grid[0, :])
    games = np.asarray(history["game_frame"], dtype=float)
    scores = np.asarray(history["game_score"], dtype=float)

    ax.scatter(games, scores, s=7, color=color, alpha=0.30, linewidths=0, zorder=2,
               label="one training game")
    if scores.size:
        ax.plot(games, rolling(scores, 100), color=color, linewidth=2.0, zorder=4,
                label="training, 100-game mean")

    ev_x = np.asarray(history["eval_frame"], dtype=float)
    ev_y = np.asarray(history["eval_mean"], dtype=float)
    if ev_x.size:
        ax.plot(ev_x, ev_y, color=eval_color, linewidth=2.0, zorder=5,
                marker="o", markersize=4.5, markeredgecolor=vs.SURFACE, markeredgewidth=1.2,
                label="greedy evaluation, 10-game mean")
        best = int(np.argmax(ev_y))
        # Flip the callout inward when the peak is the last point, otherwise
        # it runs off the right edge.
        near_end = ev_x[best] > ev_x[0] + 0.82 * (ev_x[-1] - ev_x[0])
        ax.annotate(f"best {ev_y[best]:.1f}",
                    xy=(ev_x[best], ev_y[best]),
                    xytext=(-8 if near_end else 8, 8), textcoords="offset points",
                    color=vs.INK, fontsize=9.5, fontweight="bold",
                    ha="right" if near_end else "left", va="bottom")

    ax.axhline(20, color=vs.STATUS_GOOD, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("full-credit target: 20", xy=(0.995, 20), xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points",
                color=vs.STATUS_GOOD, fontsize=8.5, ha="right", va="bottom")

    ax.set_xlabel("environment frames")
    ax.set_ylabel("score (hits before a miss)")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(thousands)
    # Legend above the axes: the evaluation curve runs right along the top.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncols=3,
              handletextpad=0.6, columnspacing=1.6)
    vs.strip_spines(ax)

    # -- three diagnostic panels --------------------------------------------
    panels = [
        ("TD loss (Huber)", history.get("loss", []), color, True),
        ("mean Q of taken action", history.get("q_mean", []), vs.SERIES[2], False),
        ("exploration rate", history.get("epsilon", []), vs.SERIES[3], False),
    ]
    frames = np.asarray(history.get("loss_frame", []), dtype=float)
    for col, (label, series, hue, log) in enumerate(panels):
        sub = fig.add_subplot(grid[1, col])
        values = np.asarray(series, dtype=float)
        good = np.isfinite(values)
        if good.any():
            sub.plot(frames[good], rolling(values[good], 25), color=hue, linewidth=1.8)
        if log:
            sub.set_yscale("log")
        sub.set_title(label, fontsize=10, color=vs.INK_2)
        sub.set_xlabel("frames")
        sub.xaxis.set_major_formatter(thousands)
        vs.strip_spines(sub)

    final = history.get("final_mean", float("nan"))
    vs.title_block(
        fig,
        f"Deep-Q pong — {arch} backbone",
        f"{history['parameters']:,} parameters · {len(scores):,} training games · "
        f"final greedy average {final:.1f} over {len(history.get('final_scores', []))} games",
    )
    return save(fig, name)


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #


def true_skill(arch: str) -> Optional[Dict]:
    """The large-budget evaluate.py summary for this backbone, if one exists."""
    hits = sorted(RUNS_DIR.glob(f"eval_{arch}_*.json"))
    if not hits:
        return None
    best = max((json.loads(p.read_text()) for p in hits),
               key=lambda r: r.get("frame_budget", 0))
    return best


def figure_compare(histories: List[Dict], name: str = "fig_architectures.png") -> Path:
    # Rank by the large-budget score when it is available: the training loop's
    # own final mean is capped, and the cap does not merely compress the scale,
    # it reorders the backbones.
    for h in histories:
        h["skill"] = true_skill(h["arch"])
    scored = all(h["skill"] for h in histories)
    key = (lambda h: -h["skill"]["mean"]) if scored else (lambda h: -h["final_mean"])
    histories = sorted(histories, key=key)
    n = len(histories)

    fig = plt.figure(figsize=(11, 8.2))
    grid = GridSpec(2, max(2, n), figure=fig, height_ratios=[1.35, 1],
                    hspace=0.5, wspace=0.24, top=0.85, bottom=0.09, left=0.065, right=0.975)

    # -- small multiples: one panel per backbone, the others ghosted behind --
    for col, hist in enumerate(histories):
        ax = fig.add_subplot(grid[0, col])
        arch = hist["arch"]
        for other in histories:
            if other is hist:
                continue
            ax.plot(other["eval_frame"], other["eval_mean"], color=vs.GRID,
                    linewidth=1.2, zorder=2)
        ax.plot(hist["eval_frame"], hist["eval_mean"],
                color=vs.SERIES_BY_NAME[arch], linewidth=2.2, zorder=4,
                dashes=vs.DASH_BY_NAME[arch])
        if hist["eval_frame"]:
            # Mark the peak, not the last point: the bar chart below already
            # reports where each backbone finished.
            peak = int(np.argmax(hist["eval_mean"]))
            ax.annotate(f"best {hist['eval_mean'][peak]:.0f}",
                        xy=(hist["eval_frame"][peak], hist["eval_mean"][peak]),
                        xytext=(0, 7), textcoords="offset points",
                        color=vs.INK, fontsize=9, fontweight="bold",
                        ha="center", va="bottom")
        ax.axhline(20, color=vs.STATUS_GOOD, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.set_title(f"{arch}  ·  {hist['parameters'] / 1000:.0f}k params", fontsize=10)
        ax.set_xlabel("frames")
        if col == 0:
            ax.set_ylabel("greedy 10-game mean")
        ax.xaxis.set_major_formatter(thousands)
        vs.strip_spines(ax)

    ylim = max((max(h["eval_mean"]) for h in histories if h["eval_mean"]), default=1)
    for ax in fig.axes:
        ax.set_ylim(0, ylim * 1.24)  # headroom for the "best" callout

    # -- final scores, ranked ------------------------------------------------
    bar_ax = fig.add_subplot(grid[1, :])
    labels, values = [], []
    for row, hist in enumerate(reversed(histories)):
        arch = hist["arch"]
        value = hist["skill"]["mean"] if scored else hist["final_mean"]
        capped = hist["skill"]["capped_games"] if scored else 0
        label = f"{value:,.0f}" if value >= 1000 else f"{value:.1f}"
        if capped:
            label += " +"   # games that ran out of budget: a lower bound
        if scored:
            dot_row(bar_ax, row, value, vs.SERIES_BY_NAME[arch], left=1.0)
            bar_ax.annotate(label, xy=(value, row), xytext=(0, 11),
                            textcoords="offset points", color=vs.INK, fontsize=9.5,
                            fontweight="bold", ha="center", va="bottom")
        else:
            rounded_bar(bar_ax, row, value, vs.SERIES_BY_NAME[arch])
            bar_ax.text(value, row + 0.36, label, color=vs.INK, fontsize=9.5,
                        fontweight="bold", va="center", ha="right")
        labels.append(arch)
        values.append(value)

    bar_ax.axvline(20, color=vs.STATUS_GOOD, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    bar_ax.text(20, len(histories) - 0.35, " full-credit target",
                color=vs.STATUS_GOOD, fontsize=8.5, va="center")
    bar_ax.set_yticks(range(len(labels)), labels, color=vs.INK_2, fontsize=10)
    if scored:
        bar_ax.set_xscale("log")
        bar_ax.set_xlim(max(1.0, min(values) * 0.35), max(values) * 3.0)
    else:
        bar_ax.set_xlim(0, max(values + [20]) * 1.16)
    bar_ax.set_ylim(-0.6, len(labels) - 0.25)
    bar_ax.set_xlabel(
        "mean score over greedy games, log scale   (frame budget per game set "
        "large enough that games end on a miss; + = some hit the budget anyway)"
        if scored else "final greedy average score")
    bar_ax.grid(axis="y", visible=False)
    vs.strip_spines(bar_ax, keep=("bottom",))

    vs.title_block(
        fig,
        "Which backbone plays pong best?",
        "Same Double-DQN, replay buffer and schedule; only the network differs. "
        "Top: the capped evaluation the training loop runs. Bottom: what the "
        "finished models really score, on a log scale.",
    )
    return save(fig, name)


# --------------------------------------------------------------------------- #
# policy / Q landscape
# --------------------------------------------------------------------------- #


def _grid_states(ball_x, ball_y, vx, vy, paddle_y) -> np.ndarray:
    """Cartesian product of the five state variables, flattened to (N, 5)."""
    mesh = np.meshgrid(ball_x, ball_y, vx, vy, paddle_y, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1).astype(np.float32)


def typical_velocity(agent, frames: int = 3000) -> tuple:
    """Median |vx|, |vy| the agent actually encounters while playing.

    A policy map is a 2-D slice through a 5-D state space, and the slice has to
    be taken somewhere the agent actually goes. The ball speeds up on every hit
    (x1.1, capped at 8), so a strong agent spends almost all its time near the
    speed limit -- asking it about the spawn speed of 4 is asking about states
    it has effectively never seen, and the answer looks arbitrary.
    """
    sample = record_rollout(agent, max_frames=frames)
    pool = [f for f in sample if f["vx"] > 0] or sample
    vx = float(np.median([abs(f["vx"]) for f in pool]))
    vy = float(np.median([abs(f["vy"]) for f in pool]))
    return max(1.0, round(vx)), max(1.0, round(vy))


def figure_policy(agent, name: str = "fig_policy.png", velocity=None) -> Path:
    nx, ny = 120, 90
    xs = np.linspace(BALL_R, GAME_W - BALL_R, nx)
    ys = np.linspace(BALL_R, GAME_H - BALL_R, ny)

    vx, vy = velocity if velocity else typical_velocity(agent)

    # Panel A/B: the ball approaching the paddle (vx > 0), paddle at mid-board.
    states = _grid_states(xs, ys, [vx], [vy], [GAME_H / 2])
    q = agent.q_values_for_states(states).reshape(nx, ny, 3)
    value = q.max(axis=2).T
    policy = q.argmax(axis=2).T
    # How strongly the choice is held: best minus runner-up. Far from the
    # paddle the ball still has several wall bounces to go, each one randomised,
    # so the three actions are nearly tied and the argmax is close to arbitrary.
    ordered = np.sort(q, axis=2)
    margin = (ordered[:, :, 2] - ordered[:, :, 1]).T

    # Panel C: does it chase the ball? Q(down) - Q(up) over ball_y x paddle_y.
    ny2 = 110
    ball_ys = np.linspace(BALL_R, GAME_H - BALL_R, ny2)
    paddle_ys = np.linspace(PADDLE_H / 2, GAME_H - PADDLE_H / 2, ny2)
    states_c = _grid_states([GAME_W - 120], ball_ys, [vx], [0.0], paddle_ys)
    q_c = agent.q_values_for_states(states_c).reshape(ny2, ny2, 3)
    chase = (q_c[:, :, 2] - q_c[:, :, 0]).T  # down minus up

    fig = plt.figure(figsize=(12.6, 4.5))
    grid = GridSpec(1, 3, figure=fig, wspace=0.26, top=0.76, bottom=0.14, left=0.05, right=0.965)

    ax_a = fig.add_subplot(grid[0, 0])
    mesh = ax_a.pcolormesh(xs, ys, value, cmap=vs.SEQUENTIAL, shading="gouraud", rasterized=True)
    bar = fig.colorbar(mesh, ax=ax_a, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(color=vs.MUTED, labelcolor=vs.MUTED, labelsize=8)
    bar.set_label("max over actions of Q", color=vs.MUTED, fontsize=8.5)
    ax_a.set_title("value of the board", fontsize=10.5)
    ax_a.set_xlabel("ball x (px)")
    ax_a.set_ylabel("ball y (px)")
    ax_a.invert_yaxis()  # pong's y axis points down

    ax_b = fig.add_subplot(grid[0, 1])
    # Hue is the action, opacity is the margin: washed-out regions are ones
    # where the network barely prefers what it picked.
    rgba = vs.POLICY_CMAP(policy / 2.0)
    scale = np.percentile(margin, 95) or 1.0
    rgba[..., 3] = np.clip(margin / scale, 0.12, 1.0)
    ax_b.imshow(rgba, origin="lower", aspect="auto",
                extent=(xs[0], xs[-1], ys[0], ys[-1]), interpolation="nearest")
    ax_b.set_title("chosen action, faded where it is a near-tie", fontsize=10.5)
    ax_b.set_xlabel("ball x (px)")
    ax_b.invert_yaxis()
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          markerfacecolor=vs.ACTION_COLORS[a], markeredgecolor=vs.SURFACE,
                          label=vs.ACTION_LABELS[a]) for a in (-1, 0, 1)]
    ax_b.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=3)

    ax_c = fig.add_subplot(grid[0, 2])
    limit = float(np.abs(chase).max()) or 1.0
    mesh_c = ax_c.pcolormesh(ball_ys, paddle_ys, chase, cmap=vs.DIVERGING,
                             vmin=-limit, vmax=limit, shading="gouraud", rasterized=True)
    lo = max(ball_ys[0], paddle_ys[0])
    hi = min(ball_ys[-1], paddle_ys[-1])
    ax_c.plot([lo, hi], [lo, hi], color=vs.INK, linewidth=1.0,
              linestyle=(0, (3, 3)), alpha=0.65)
    ax_c.annotate("paddle level with ball", xy=(hi, hi), xytext=(-6, -10),
                  textcoords="offset points", color=vs.INK_2, fontsize=8.5,
                  ha="right", va="top")
    ax_c.set_xlim(ball_ys[0], ball_ys[-1])
    ax_c.set_ylim(paddle_ys[0], paddle_ys[-1])
    bar_c = fig.colorbar(mesh_c, ax=ax_c, pad=0.02)
    bar_c.outline.set_visible(False)
    bar_c.ax.tick_params(color=vs.MUTED, labelcolor=vs.MUTED, labelsize=8)
    bar_c.set_label("Q(down) − Q(up)", color=vs.MUTED, fontsize=8.5)
    ax_c.set_title("does it chase the ball?", fontsize=10.5)
    ax_c.set_xlabel("ball y (px)")
    ax_c.set_ylabel("paddle y (px)")

    for ax in (ax_a, ax_b, ax_c):
        ax.grid(False)
        vs.strip_spines(ax, keep=())

    vs.title_block(
        fig,
        "What the trained network believes",
        f"{agent.config.arch} backbone, sliced at the speeds it actually plays at "
        f"(vx=+{vx:.0f}, vy=+{vy:.0f}, the medians over a greedy rollout). Left two panels "
        "hold the paddle at mid-board; the right one sweeps both heights at vy=0.",
        y=0.975,
    )
    return save(fig, name)


# --------------------------------------------------------------------------- #
# rollout animation
# --------------------------------------------------------------------------- #


def record_rollout(agent, max_frames: int = 900, paddle_speed: float = 8.0,
                   ball_speed: float = 4.0, warmup: int = 0) -> List[Dict]:
    """Play greedily and record everything the animation needs.

    ``warmup`` frames are played but not recorded. The ball spawns at speed 4
    and only ramps to the cap of 8 over the first few rallies, so the opening
    of a game runs at ~300 frames per hit against ~150 later; skipping it is
    the difference between a clip with four rallies in it and one with forty.
    """
    import pong

    agent.eval_mode()
    game = pong.PongGame(ball_speed=ball_speed, paddle_speed=paddle_speed,
                         learner=agent, visible=False, state_quantization=None)
    state = game.state_init()
    frames: List[Dict] = []

    for _ in range(warmup):
        action = agent.act(state)
        newstate, reward = game.update(state, action * paddle_speed)
        agent.learn(state, action, reward, newstate)
        state = newstate

    for _ in range(max_frames):
        q = np.asarray(agent.report_q(state), dtype=float)
        action = agent.act(state)
        frames.append({
            "ball": (state[0], state[1]),
            "vx": state[2],
            "vy": state[3],
            "paddle": state[4],
            "score": game.score,
            "q": q,
            "action": action,
        })
        newstate, reward = game.update(state, action * paddle_speed)
        agent.learn(state, action, reward, newstate)
        state = newstate

    return frames


def figure_rollout(agent, name: Optional[str] = None, max_frames: int = 600,
                   stride: int = 3, fps: int = 20, dpi: int = 100,
                   video: bool = False, warmup: int = 0) -> Path:
    """Animate a greedy rally.

    ``video=True`` writes an H.264 mp4 instead of a GIF: far better quality per
    byte, which is what makes a minute of play at 50 fps practical. A GIF of
    the same length would be tens of megabytes.
    """
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    if name is None:
        name = "rollout.mp4" if video else "rollout.gif"

    recorded = record_rollout(agent, max_frames=max_frames, warmup=warmup)
    scores = [f["score"] for f in recorded]
    hits = sum(1 for a, b in zip(scores, scores[1:]) if b > a)
    misses = sum(1 for a, b in zip(scores, scores[1:]) if b < a)
    frames = recorded[::stride]
    trail_len = 20 if video else 14

    # The board is drawn with equal aspect (600x400), so the panel it sits in
    # has to be close to 1.5:1 or the frame fills up with empty margin.
    fig = plt.figure(figsize=(12.8, 6.6) if video else (9.0, 3.9))
    grid = GridSpec(1, 2, figure=fig,
                    width_ratios=[1.5, 1] if video else [2.5, 1], wspace=0.18,
                    top=0.80 if video else 0.74, bottom=0.10 if video else 0.12,
                    left=0.03, right=0.95)

    # -- the board -----------------------------------------------------------
    board = fig.add_subplot(grid[0, 0])
    board.set_xlim(0, GAME_W)
    board.set_ylim(GAME_H, 0)  # pong's y axis points down
    board.set_aspect("equal")
    board.grid(False)
    board.set_xticks([])
    board.set_yticks([])
    for spine in board.spines.values():
        spine.set_visible(True)
        spine.set_color(vs.AXIS)
    board.axvline(GAME_W / 2, color=vs.GRID, linewidth=1.0)

    trail = board.scatter([], [], s=[], color=vs.SERIES[1], alpha=0.35, linewidths=0, zorder=3)
    ball = plt.Circle((0, 0), BALL_R * (1.15 if video else 1.0), color=vs.SERIES[1], zorder=5)
    board.add_patch(ball)
    paddle_w = PADDLE_W * (2.0 if video else 1.0)   # 8px is invisible at video size
    paddle = plt.Rectangle((GAME_W - paddle_w, 0), paddle_w, PADDLE_H,
                           color=vs.SERIES[0], zorder=5)
    board.add_patch(paddle)
    # Above the board, not inside it: the ball reaches every corner.
    score_text = board.text(0.0, 1.03, "", transform=board.transAxes, color=vs.INK,
                            fontsize=12, fontweight="bold", va="bottom", ha="left")

    # -- the live Q panel ----------------------------------------------------
    q_ax = fig.add_subplot(grid[0, 1])
    actions = [-1, 0, 1]
    bars = []
    for row, act in enumerate(actions):
        line, = q_ax.plot([0, 0], [row, row], linewidth=13, solid_capstyle="round",
                          color=vs.SERIES[0], zorder=3)
        bars.append(line)
    labels = [q_ax.text(0, row + 0.30, "", color=vs.INK_2, fontsize=9.5, va="center")
              for row in range(3)]
    q_ax.set_yticks(range(3), [vs.ACTION_LABELS[a] for a in actions], color=vs.INK_2, fontsize=10)
    q_ax.set_ylim(2.6, -0.6)  # "up" on top, to match the paddle on the board
    q_ax.set_xlabel("Q value")
    q_ax.set_title("action values", fontsize=10.5, pad=6)
    q_ax.grid(axis="y", visible=False)
    vs.strip_spines(q_ax, keep=("bottom",))

    q_all = np.stack([f["q"] for f in frames])
    q_lo, q_hi = float(q_all.min()), float(q_all.max())
    baseline = min(0.0, q_lo)          # a bar chart has to start at zero
    span = max(1e-3, q_hi - baseline)
    q_ax.set_xlim(baseline, q_hi + 0.22 * span)
    q_ax.axvline(0, color=vs.AXIS, linewidth=1.0, zorder=2)

    speed = f"{stride}x speed" if stride > 1 else "real time"
    outcome = (f"{hits} rallies, {misses} misses" if misses
               else f"{hits} consecutive rallies without a miss")
    vs.title_block(
        fig,
        "One greedy rally",
        f"{agent.config.arch} backbone, exploration off: {outcome} ({speed}"
        + (f", after a {warmup:,}-frame warm-up" if warmup else "")
        + "). The right-hand panel is the network's action value at that "
        "instant; the highlighted bar is the action it takes.",
        y=0.97,
    )

    def draw(i: int):
        frame = frames[i]
        ball.center = frame["ball"]
        paddle.set_y(frame["paddle"] - PADDLE_H / 2)
        score_text.set_text(f"score {frame['score']}")

        window = frames[max(0, i - trail_len): i]
        if window:
            trail.set_offsets([f["ball"] for f in window])
            trail.set_sizes(np.linspace(8, 150 if video else 90, len(window)))
        else:
            trail.set_offsets(np.empty((0, 2)))
            trail.set_sizes([])

        chosen = actions.index(frame["action"])
        for row, (line, label) in enumerate(zip(bars, labels)):
            value = frame["q"][row]
            line.set_data([baseline, value], [row, row])
            line.set_color(vs.SERIES[0] if row == chosen else vs.GRID)
            label.set_text(f"{value:.2f}")
            label.set_position((value, row + 0.30))
            label.set_color(vs.INK if row == chosen else vs.MUTED)
        return [ball, paddle, trail, score_text, *bars, *labels]

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    if video:
        # yuv420p and even dimensions are what make the file play in browsers
        # and QuickTime rather than only in ffplay.
        writer = FFMpegWriter(
            fps=fps, codec="h264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"],
        )
    else:
        writer = PillowWriter(fps=fps)
    # matplotlib's animation writers already force bbox_inches off (every frame
    # has to come out the same size), so the global "tight" default is fine here.
    anim.save(path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {path.relative_to(HERE)} ({len(frames)} frames)")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def load_agent(model_path: str, device: str = "cpu"):
    import submitted

    agent = submitted.deep_q(0.05, 0.05, 0.99, 5, device=device)
    agent.load(model_path)
    return agent


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("curves", "compare", "policy", "rollout", "video", "all"))
    parser.add_argument("--run", nargs="*", default=None,
                        help="run directories (default: every run under runs/)")
    parser.add_argument("--model", default="trained_model.pkl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollout-frames", type=int, default=600)
    parser.add_argument("--stride", type=int, default=0,
                        help="env frames per rendered frame (0 = 3 for gif, 2 for video)")
    parser.add_argument("--fps", type=int, default=0, help="0 = 20 for gif, 50 for video")
    parser.add_argument("--warmup", type=int, default=-1,
                        help="env frames to play before recording (-1 = 7000 for video, 0 for gif)")
    parser.add_argument("--policy-velocity", type=float, nargs=2, default=None,
                        metavar=("VX", "VY"),
                        help="slice the policy map at these speeds (default: the medians "
                             "measured from a greedy rollout)")
    args = parser.parse_args(argv)

    vs.use_dark_style()
    want = {"curves", "compare", "policy", "rollout"} if args.what == "all" else {args.what}
    if "video" in want:
        want = {"rollout"}
        args.video = True
    else:
        args.video = False
    runs = find_runs(args.run)

    if want & {"curves", "compare"} and not runs:
        print("  no runs found under runs/ -- train first with train_deepq.py")

    if "curves" in want:
        for run_dir in runs:
            history = load_history(run_dir)
            figure_curves(history, f"fig_learning_{history['arch']}.png")

    if "compare" in want and len(runs) > 1:
        figure_compare([load_history(r) for r in runs])

    if want & {"policy", "rollout"}:
        if not Path(args.model).exists():
            print(f"  {args.model} not found -- skipping policy/rollout figures")
            return
        agent = load_agent(args.model, args.device)
        if "policy" in want:
            velocity = tuple(args.policy_velocity) if args.policy_velocity else None
            figure_policy(agent, velocity=velocity)
        if "rollout" in want:
            warmup = args.warmup if args.warmup >= 0 else (7000 if args.video else 0)
            figure_rollout(agent, max_frames=args.rollout_frames, video=args.video,
                           stride=args.stride or (2 if args.video else 3),
                           fps=args.fps or (50 if args.video else 20), warmup=warmup)


if __name__ == "__main__":
    main()
