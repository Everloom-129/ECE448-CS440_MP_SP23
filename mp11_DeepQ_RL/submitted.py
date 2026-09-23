"""MP11 -- reinforcement learning for one-player Pong.

This module holds the two learners that ``pong.PongGame`` knows how to drive:

``q_learner``
    The required part of the MP: a tabular Q-learner over the quantized
    ``[ball_x, ball_y, ball_vx, ball_vy, paddle_y]`` state.

``deep_q``
    The extra-credit part: a Double-DQN over the *unquantized* state, with a
    choice of four backbones (``mlp``, ``cnn``, ``resnet``, ``transformer``)
    selected through :class:`DeepQConfig`.

Everything the autograder needs lives in this one file -- Gradescope only
receives ``submitted.py`` and ``trained_model.pkl``, so the network
definitions cannot be imported from a sibling module.

References
----------
- Mnih et al., "Human-level control through deep reinforcement learning", 2015
  (replay buffer + target network).
- van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning",
  2016 (the ``double_dqn`` action/value split).
- Wang et al., "Dueling Network Architectures for Deep RL", 2016 (the
  value/advantage head).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Environment constants (mirrors the defaults of pong.PongGame)
# --------------------------------------------------------------------------- #

ACTIONS: Tuple[int, int, int] = (-1, 0, 1)  # paddle: up, hold, down
N_ACTIONS: int = len(ACTIONS)
STATE_DIM: int = 5  # ball_x, ball_y, ball_vx, ball_vy, paddle_y

GAME_W: float = 600.0
GAME_H: float = 400.0
MAX_BALL_SPEED: float = 8.0  # PongGame caps |v| at 2 * ball_speed, ball_speed=4

# Per-variable divisor and offset that map a raw pong state onto [-1, 1].
_STATE_SCALE = np.array(
    [GAME_W, GAME_H, MAX_BALL_SPEED, MAX_BALL_SPEED, GAME_H], dtype=np.float32
)
_STATE_SHIFT = np.array([0.5, 0.5, 0.0, 0.0, 0.5], dtype=np.float32)


def tracking_potential(state: Sequence[float], weight: float) -> float:
    """Potential for reward shaping: how well the paddle shadows the ball.

    Zero unless the ball is travelling toward the paddle, because while the
    ball is heading away the paddle's position genuinely does not matter yet.
    Scaled to the board height so the shaped term stays small next to the
    environment's own +1 / -10.

    @params:
    state (sequence of 5 floats): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
    weight (float): the shaping strength, DeepQConfig.shaping.

    @return:
    phi (float): 0 when the ball recedes, else -weight * |ball_y - paddle_y| / H.
    """
    if weight == 0.0 or state[2] <= 0:  # ball_vx <= 0: moving away from the paddle
        return 0.0
    return -weight * abs(state[1] - state[4]) / GAME_H


def normalize_state(state: Sequence[float]) -> np.ndarray:
    """Map a raw pong state onto ``[-1, 1]^5``.

    Feeding raw pixel coordinates (0..600) to a network makes the first layer
    do nothing but rescale, so normalising here is what lets a small net learn
    quickly.

    @params:
    state (sequence of 5 floats): ball_x, ball_y, ball_vx, ball_vy, paddle_y.

    @return:
    obs (float32 array of shape (5,)): the normalized state.
    """
    raw = np.asarray(state, dtype=np.float32)
    return (raw / _STATE_SCALE - _STATE_SHIFT) * 2.0


# --------------------------------------------------------------------------- #
# Part 1 (required): tabular Q-learning
# --------------------------------------------------------------------------- #


class q_learner:
    """Tabular Q-learner over the quantized pong state."""

    def __init__(self, alpha, epsilon, gamma, nfirst, state_cardinality):
        '''
        Create a new q_learner object.

        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting
        state_cardinality (list) - cardinality of each of the quantized state variables

        @return:
        None
        '''
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.nfirst = nfirst
        self.sc = state_cardinality
        self.flag = False  # set True to force pure exploitation

        # Q[...state..., action] = expected utility; N[...] = exploration count.
        # Actions -1/0/+1 are stored at indices 0/1/2.
        self.Q = np.zeros((*state_cardinality, N_ACTIONS))
        self.N = np.zeros((*state_cardinality, N_ACTIONS))

    def report_exploration_counts(self, state):
        '''
        Check to see how many times each action has been explored in this state.

        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.

        @return:
        explored_count (array of 3 ints):
          number of times that each action has been explored from this state.
        '''
        return self.N[tuple(state)]

    def choose_unexplored_action(self, state):
        '''
        Choose an action that has been explored less than nfirst times.

        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.

        @return:
        action (scalar): either -1, or 0, or 1, or None
          None if every action has already been explored nfirst times;
          otherwise one chosen uniformly from the under-explored actions,
          whose count is incremented.
        '''
        explored_count = self.report_exploration_counts(state)
        underexplored = [i for i, count in enumerate(explored_count) if count < self.nfirst]
        if not underexplored:
            return None

        chosen_idx = random.choice(underexplored)
        self.N[tuple(state)][chosen_idx] += 1
        return ACTIONS[chosen_idx]

    def report_q(self, state):
        '''
        Report the current Q values for the given state.

        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.

        @return:
        Q (array of 3 floats): the Q value of each of the three actions.
        '''
        return self.Q[tuple(state)]

    def q_local(self, reward, newstate):
        '''
        The update to Q estimated from a single step of game play:
        reward plus gamma times the max of Q[newstate, ...].

        @params:
        reward (scalar float): the reward achieved from the current step of game play.
        newstate (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.

        @return:
        Q_local (scalar float): the local value of Q
        '''
        return reward + self.gamma * np.max(self.report_q(newstate))

    def learn(self, state, action, reward, newstate):
        '''
        Update the internal Q-table on the basis of an observed
        state, action, reward, newstate sequence.

        @params:
        state: a list of 5 ints, the quantized state before the move
        action: an integer, one of -1, 0, or +1
        reward: positive for hitting the ball, negative for losing a game
        newstate: a list of 5 ints, the quantized state after the move

        @return:
        None
        '''
        index = tuple(state) + (ACTIONS.index(action),)
        q_current = self.Q[index]
        self.Q[index] = q_current + self.alpha * (self.q_local(reward, newstate) - q_current)

    def save(self, filename):
        '''
        Save the Q and N tables to ``filename`` in numpy's ``.npz`` format.
        '''
        np.savez(filename, Q=self.Q, N=self.N)

    def load(self, filename):
        '''
        Load the Q and N tables from a file written by :meth:`save`.
        '''
        data = np.load(filename)
        self.Q = data["Q"]
        self.N = data["N"]

    def exploit(self, state):
        '''
        Return the action with the highest Q-value for this state, and that Q-value.

        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.

        @return:
        action (scalar int): -1, 0, or 1
        Q (scalar float): the Q-value of the selected action
        '''
        q = self.report_q(state)
        best_idx = int(np.argmax(q))
        return ACTIONS[best_idx], q[best_idx]

    def act(self, state):
        '''
        Decide what action to take in the current state: explore any
        under-explored action first, then epsilon-greedy over the Q-table.

        @params:
        state: a list of 5 ints, the quantized state

        @return:
        -1 to move the paddle up, 0 to hold, +1 to move down
        '''
        action = self.choose_unexplored_action(state)
        if action is not None:
            return action
        if not self.flag and random.random() <= self.epsilon:
            return random.choice(ACTIONS)
        action, _ = self.exploit(state)
        return action


# --------------------------------------------------------------------------- #
# Part 2 (extra credit): network architectures
#
# Every backbone consumes a stack of the last ``n_frames`` normalized states,
# shaped (batch, n_frames, 5), and emits a feature vector (batch, out_dim).
# Stacking is what gives the convolutional and attention backbones a genuine
# sequence axis to work on: a single 5-number state has no structure to
# convolve over, but the recent trajectory does.
# --------------------------------------------------------------------------- #


class MLPBackbone(nn.Module):
    """Flatten the frame stack and run it through a LayerNorm-ed MLP."""

    def __init__(self, n_frames: int, width: int = 128, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        d_in = n_frames * STATE_DIM
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.LayerNorm(width), nn.SiLU()]
            d_in = width
        self.net = nn.Sequential(*layers)
        self.out_dim = width

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.flatten(1))


class CNNBackbone(nn.Module):
    """1-D convolutions over the time axis (state variables are channels)."""

    def __init__(self, n_frames: int, channels: int = 64, depth: int = 2, kernel: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        c_in = STATE_DIM
        for _ in range(depth):
            layers += [
                nn.Conv1d(c_in, channels, kernel, padding=kernel // 2),
                nn.GroupNorm(_groups_for(channels), channels),
                nn.SiLU(),
            ]
            c_in = channels
        self.net = nn.Sequential(*layers)
        self.out_dim = 2 * channels  # mean-pool and max-pool are concatenated

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.net(obs.transpose(1, 2))  # (B, C, n_frames)
        return torch.cat([feat.mean(dim=-1), feat.amax(dim=-1)], dim=-1)


class ResidualBlock1d(nn.Module):
    """Pre-activation residual block, the ResNet-v2 ordering."""

    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        groups = _groups_for(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class ResNetBackbone(nn.Module):
    """A stack of pre-activation residual blocks over the frame axis."""

    def __init__(self, n_frames: int, channels: int = 64, blocks: int = 3, kernel: int = 3):
        super().__init__()
        self.stem = nn.Conv1d(STATE_DIM, channels, kernel_size=1)
        self.blocks = nn.Sequential(*[ResidualBlock1d(channels, kernel) for _ in range(blocks)])
        self.norm = nn.GroupNorm(_groups_for(channels), channels)
        self.out_dim = 2 * channels

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = F.silu(self.norm(self.blocks(self.stem(obs.transpose(1, 2)))))
        return torch.cat([feat.mean(dim=-1), feat.amax(dim=-1)], dim=-1)


class TransformerBackbone(nn.Module):
    """Treat the frame stack as a short token sequence and attend over it.

    Each of the ``n_frames`` states is embedded as one token; a learned CLS
    token collects the summary, exactly as in ViT/BERT.
    """

    def __init__(
        self,
        n_frames: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Linear(STATE_DIM, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_frames + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN: trains without a warmup schedule
        )
        # norm_first=True rules out the nested-tensor fast path; saying so
        # explicitly keeps torch from warning about it on every build.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(obs)  # (B, n_frames, d_model)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos
        return self.norm(self.encoder(tokens)[:, 0])


BACKBONES = {
    "mlp": MLPBackbone,
    "cnn": CNNBackbone,
    "resnet": ResNetBackbone,
    "transformer": TransformerBackbone,
}


def _groups_for(channels: int) -> int:
    """Largest GroupNorm group count <= 8 that divides ``channels``."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DuelingHead(nn.Module):
    """Split the features into a state value and per-action advantages."""

    def __init__(self, in_dim: int, hidden: int = 64, n_actions: int = N_ACTIONS):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.advantage = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, n_actions)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        advantage = self.advantage(feat)
        return self.value(feat) + advantage - advantage.mean(dim=-1, keepdim=True)


class QNetwork(nn.Module):
    """Backbone + Q-head: (batch, n_frames, 5) -> (batch, 3) action values."""

    def __init__(
        self,
        arch: str = "resnet",
        n_frames: int = 4,
        dueling: bool = True,
        head_hidden: int = 64,
        **backbone_kwargs: Any,
    ):
        super().__init__()
        if arch not in BACKBONES:
            raise ValueError(f"unknown arch {arch!r}; choose from {sorted(BACKBONES)}")
        self.arch = arch
        self.n_frames = n_frames
        self.backbone = BACKBONES[arch](n_frames, **backbone_kwargs)
        if dueling:
            self.head: nn.Module = DuelingHead(self.backbone.out_dim, head_hidden)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.backbone.out_dim, head_hidden),
                nn.SiLU(),
                nn.Linear(head_hidden, N_ACTIONS),
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(obs))


# --------------------------------------------------------------------------- #
# Part 2 (extra credit): the Double-DQN agent
# --------------------------------------------------------------------------- #


@dataclass
class DeepQConfig:
    """Everything about :class:`deep_q` that is not part of the fixed MP API.

    The defaults are the ones used to train the shipped ``trained_model.pkl``.
    """

    # --- architecture ---
    arch: str = "resnet"  # one of BACKBONES
    n_frames: int = 4  # how many past states are stacked into one observation
    dueling: bool = True
    head_hidden: int = 64
    backbone_kwargs: Dict[str, Any] = field(default_factory=dict)

    # --- optimisation ---
    lr: float = 1e-3  # Adam step size; see the note in deep_q.__init__
    batch_size: int = 256
    buffer_size: int = 200_000
    warmup: int = 5_000  # frames of pure exploration before the first update
    train_every: int = 4  # gradient steps are taken every N frames
    target_sync: int = 1_000  # frames between hard target-network copies
    grad_clip: float = 10.0
    double_dqn: bool = True
    huber_delta: float = 1.0
    n_step: int = 1  # multi-step returns; 1 is plain one-step Q-learning
    # Potential-based shaping weight (Ng, Harada & Russell 1999). 0 disables it.
    # The potential is a function of state alone and is defined to be 0 at a
    # terminal, so the optimal policy is provably unchanged -- it only makes
    # the "get behind the ball" signal dense instead of arriving 150 frames late.
    shaping: float = 0.0

    # --- exploration ---
    eps_start: float = 1.0  # decays linearly to deep_q.epsilon
    eps_decay_steps: int = 150_000

    # --- misc ---
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:1" ...
    seed: Optional[int] = None

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-capacity ring buffer of (obs, action, reward, next_obs, done)."""

    def __init__(self, capacity: int, n_frames: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_frames, STATE_DIM), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        # gamma ** (number of rewards folded in), so n-step and truncated
        # transitions bootstrap with the right factor.
        self.discount = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        obs: np.ndarray,
        action_idx: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        discount: float,
    ) -> None:
        i = self.cursor
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.action[i] = action_idx
        self.reward[i] = reward
        self.done[i] = float(done)
        self.discount[i] = discount
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.action[idx],
            self.reward[idx],
            self.next_obs[idx],
            self.done[idx],
            self.discount[idx],
        )


class deep_q:
    """Double-DQN player for ``pong.PongGame(state_quantization=None)``.

    The five methods the MP requires -- ``__init__``, ``act``, ``learn``,
    ``save``, ``load`` -- keep their documented signatures; ``report_q`` is
    also provided because ``pong.PongGame.run`` calls it to collect Q-value
    traces.

    Two behaviours are worth calling out because they are policy decisions,
    not implementation details:

    * :meth:`load` puts the learner in **evaluation mode** -- greedy actions
      and no gradient updates. Loading a checkpoint means "deploy this
      model", and the autograder plays 10 games straight after loading, so a
      5% random-action rate there would only add noise.
    * ``alpha`` is stored because the MP API asks for it, but Adam's step size
      is :attr:`DeepQConfig.lr`. A tabular learning rate of 0.05 is far too
      large for a neural optimiser.
    """

    def __init__(
        self,
        alpha: float,
        epsilon: float,
        gamma: float,
        nfirst: int,
        config: Optional[DeepQConfig] = None,
        **overrides: Any,
    ):
        '''
        Create a new deep_q learner.

        @params:
        alpha (scalar) - learning rate of the Q-learner (kept for API
          compatibility; the optimiser uses config.lr)
        epsilon (scalar) - the floor of the linear exploration schedule
        gamma (scalar) - discount factor
        nfirst (scalar) - kept for API compatibility; config.warmup is the
          number of frames explored before the first gradient step
        config (DeepQConfig) - architecture and optimiser settings
        **overrides - individual DeepQConfig fields, e.g. ``arch="transformer"``

        @return:
        None
        '''
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.nfirst = nfirst

        self.config = config or DeepQConfig()
        for key, value in overrides.items():
            if not hasattr(self.config, key):
                raise TypeError(f"unknown DeepQConfig field {key!r}")
            setattr(self.config, key, value)

        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

        self.device = self.config.resolved_device()
        self.model = self._build_model().to(self.device)
        self.target = self._build_model().to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.buffer_size, self.config.n_frames)

        self.training = True
        self.steps = 0  # frames seen by learn()
        self.updates = 0  # gradient steps taken
        self.metrics: Dict[str, float] = {"loss": float("nan"), "td_error": float("nan")}

        self._frames: Deque[np.ndarray] = deque(maxlen=self.config.n_frames)
        self._pending: Deque[tuple] = deque(maxlen=max(1, self.config.n_step))
        self._last_obs: Optional[np.ndarray] = None
        self._last_action_idx: int = ACTIONS.index(0)

        # ``self.flag`` mirrors q_learner: set it True to force pure exploitation.
        self.flag = False

    # -- construction helpers ------------------------------------------------

    def _build_model(self) -> QNetwork:
        cfg = self.config
        return QNetwork(
            arch=cfg.arch,
            n_frames=cfg.n_frames,
            dueling=cfg.dueling,
            head_hidden=cfg.head_hidden,
            **cfg.backbone_kwargs,
        )

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters in the online network."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # -- frame-stack bookkeeping --------------------------------------------

    def _reset_frames(self) -> None:
        self._frames.clear()
        self._last_obs = None

    def _fold_pending(self, flush: bool = False) -> None:
        """Turn queued one-step transitions into n-step transitions.

        The oldest queued transition is emitted once ``n_step`` rewards have
        accumulated behind it; at the end of an episode ``flush`` drains what
        is left, each with however many rewards it managed to collect.
        """
        n = max(1, self.config.n_step)
        while self._pending and (flush or len(self._pending) == n):
            obs, action_idx, _, _, _ = self._pending[0]
            total, discount, done, next_obs = 0.0, 1.0, False, self._pending[0][3]
            for _, _, reward, follow_obs, terminal in self._pending:
                total += discount * reward
                discount *= self.gamma
                next_obs = follow_obs
                if terminal:
                    done = True
                    break
            self.buffer.push(obs, action_idx, total, next_obs, done, discount)
            self._pending.popleft()

    def _push(self, state: Sequence[float]) -> np.ndarray:
        """Append a raw state to the stack and return the current observation."""
        obs = normalize_state(state)
        if not self._frames:  # start of an episode: pad by repeating
            for _ in range(self.config.n_frames):
                self._frames.append(obs)
        else:
            self._frames.append(obs)
        return np.stack(self._frames, axis=0)

    def _peek_next(self, newstate: Sequence[float]) -> np.ndarray:
        """The observation that will follow ``newstate``, without mutating state."""
        frames = list(self._frames)[1:] + [normalize_state(newstate)]
        return np.stack(frames, axis=0)

    # -- exploration schedule ------------------------------------------------

    @property
    def epsilon_now(self) -> float:
        """Current exploration rate: linear from ``eps_start`` down to ``epsilon``."""
        if not self.training or self.flag:
            return 0.0
        progress = min(1.0, self.steps / max(1, self.config.eps_decay_steps))
        return self.config.eps_start + progress * (self.epsilon - self.config.eps_start)

    # -- the MP API ----------------------------------------------------------

    def act(self, state):
        '''
        Decide what action to take in the current state.

        Epsilon-greedy while training (epsilon decays linearly from
        config.eps_start to self.epsilon); greedy once :meth:`load` or
        :meth:`eval_mode` has been called.

        @params:
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y.

        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        '''
        obs = self._push(state)
        self._last_obs = obs

        if random.random() < self.epsilon_now:
            action_idx = random.randrange(N_ACTIONS)
        else:
            action_idx = int(np.argmax(self._forward(obs[None, ...])[0]))

        self._last_action_idx = action_idx
        return ACTIONS[action_idx]

    def report_q(self, state):
        '''
        Report the current Q values for the given state.

        @params:
        state (list of 5 floats): ball_x, ball_y, ball_vx, ball_vy, paddle_y.

        @return:
        Q (array of 3 floats): the value of each of the three actions.
        '''
        current = normalize_state(state)
        if not self._frames:
            obs = np.repeat(current[None, :], self.config.n_frames, axis=0)
        elif np.array_equal(self._frames[-1], current):
            # act() already pushed this state; report on exactly what it saw.
            obs = np.stack(self._frames, axis=0)
        else:
            obs = np.stack(list(self._frames)[1:] + [current], axis=0)
        return self._forward(obs[None, ...])[0]

    def learn(self, state, action, reward, newstate):
        '''
        Perform one iteration of training on a deep-Q model.

        Stores the transition in the replay buffer and, every
        ``config.train_every`` frames past ``config.warmup``, takes one
        Double-DQN gradient step. A negative reward means the ball was
        missed, which both ends the episode (no bootstrapping) and clears the
        frame stack.

        In evaluation mode this only maintains the frame stack.

        @params:
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y
        action: an integer, one of -1, 0, or +1
        reward: positive for hitting the ball, negative for losing a game
        newstate: a list of 5 floats, in the same format as state

        @return:
        None
        '''
        terminal = reward < 0

        if not self.training:
            if terminal:
                self._reset_frames()
            return

        obs = self._last_obs if self._last_obs is not None else self._push(state)
        next_obs = self._peek_next(newstate)
        action_idx = ACTIONS.index(action) if action in ACTIONS else self._last_action_idx
        shaped = float(reward)
        if self.config.shaping:
            # F = gamma * phi(s') - phi(s), with phi(terminal) := 0.
            phi_next = 0.0 if terminal else tracking_potential(newstate, self.config.shaping)
            shaped += self.gamma * phi_next - tracking_potential(state, self.config.shaping)
        self._pending.append((obs, action_idx, shaped, next_obs, terminal))
        self._fold_pending(flush=terminal)

        self.steps += 1
        if terminal:
            self._reset_frames()

        if len(self.buffer) >= max(self.config.batch_size, self.config.warmup):
            if self.steps % self.config.train_every == 0:
                self._optimize()
            if self.steps % self.config.target_sync == 0:
                self.target.load_state_dict(self.model.state_dict())

    def save(self, filename):
        '''
        Save the trained deep-Q model, together with the architecture needed
        to rebuild it, to ``filename`` (``torch.save``, so a ``.pkl``
        extension is conventional here).

        @params:
        filename (str) - filename to which it should be saved

        @return:
        None
        '''
        torch.save(
            {
                "format": 1,
                "config": asdict(self.config),
                "state_dict": self.model.state_dict(),
                "hyperparameters": {
                    "alpha": self.alpha,
                    "epsilon": self.epsilon,
                    "gamma": self.gamma,
                    "nfirst": self.nfirst,
                },
                "steps": self.steps,
                "updates": self.updates,
            },
            filename,
        )

    def load(self, filename):
        '''
        Load a deep-Q model written by :meth:`save`, rebuilding the network
        from the architecture recorded in the checkpoint, and switch to
        evaluation mode (greedy actions, no further learning).

        @params:
        filename (str) - filename from which it should be loaded

        @return:
        None
        '''
        checkpoint = torch.load(filename, map_location="cpu", weights_only=False)

        saved = dict(checkpoint["config"])
        saved["device"] = self.config.device  # keep the caller's device choice
        self.config = DeepQConfig(**saved)
        self.device = self.config.resolved_device()

        self.model = self._build_model().to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.target = self._build_model().to(self.device)
        self.target.load_state_dict(checkpoint["state_dict"])

        self.buffer = ReplayBuffer(self.config.buffer_size, self.config.n_frames)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self._frames = deque(maxlen=self.config.n_frames)
        self._pending = deque(maxlen=max(1, self.config.n_step))
        self._reset_frames()
        self.eval_mode()

    def end_episode(self) -> None:
        """Close out an episode, including one cut short by a frame budget.

        Flushes whatever is still queued for the n-step return and clears the
        frame stack, so no transition can straddle the boundary between two
        episodes. Truncation is deliberately *not* treated as a terminal:
        nothing here is marked done, so the queued transitions keep
        bootstrapping from their own next state, which is what actually
        happened.
        """
        if self.training:
            self._fold_pending(flush=True)
        self._pending.clear()
        self._reset_frames()

    # -- mode switches -------------------------------------------------------

    def train_mode(self) -> "deep_q":
        """Resume epsilon-greedy exploration and gradient updates."""
        self.training = True
        self.model.train()
        return self

    def eval_mode(self) -> "deep_q":
        """Play greedily and stop learning."""
        self.training = False
        self.model.eval()
        self.target.eval()
        return self

    # -- internals -----------------------------------------------------------

    @torch.inference_mode()
    def _forward(self, obs_batch: np.ndarray) -> np.ndarray:
        """Q-values for a batch of stacked observations.

        No train()/eval() toggle here on purpose: none of the backbones use
        dropout or BatchNorm, so the mode changes nothing numerically, and
        ``Module.train()`` walks every submodule -- a cost that would land on
        every frame of the rollout.
        """
        tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        return self.model(tensor).cpu().numpy()

    def _optimize(self) -> None:
        """One Double-DQN gradient step on a replay minibatch."""
        cfg = self.config
        obs, action, reward, next_obs, done, discount = self.buffer.sample(cfg.batch_size)

        obs_t = torch.as_tensor(obs, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, device=self.device)
        action_t = torch.as_tensor(action, device=self.device)
        reward_t = torch.as_tensor(reward, device=self.device)
        done_t = torch.as_tensor(done, device=self.device)
        discount_t = torch.as_tensor(discount, device=self.device)

        q_taken = self.model(obs_t).gather(1, action_t[:, None]).squeeze(1)

        with torch.no_grad():
            if cfg.double_dqn:
                # Online net picks the action, target net scores it.
                best = self.model(next_obs_t).argmax(dim=1, keepdim=True)
                q_next = self.target(next_obs_t).gather(1, best).squeeze(1)
            else:
                q_next = self.target(next_obs_t).max(dim=1).values
            target = reward_t + discount_t * (1.0 - done_t) * q_next

        loss = F.smooth_l1_loss(q_taken, target, beta=cfg.huber_delta)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.optimizer.step()

        self.updates += 1
        self.metrics = {
            "loss": float(loss.detach()),
            "td_error": float((target - q_taken.detach()).abs().mean()),
            "q_mean": float(q_taken.detach().mean()),
        }

    # -- helpers for the visualisation scripts -------------------------------

    def q_values_for_states(self, states: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Q-values for many raw states at once, for plotting.

        Each state is expanded into a frame stack by running the ball
        *backwards* along its own velocity, so the earlier frames are where it
        actually was. Simply repeating the state would describe a ball that is
        stationary and moving at the same time -- an input the network never
        sees in play, and one it answers close to arbitrarily.

        Wall bounces are ignored in the reconstruction, which only matters
        within a few pixels of an edge. The paddle is assumed to have been
        holding still.

        @params:
        states (array of shape (N, 5)): raw, unquantized pong states.

        @return:
        Q (array of shape (N, 3)): action values under the current network.
        """
        raw = np.asarray(states, dtype=np.float32).reshape(-1, STATE_DIM)
        n_frames = self.config.n_frames
        history = np.repeat(raw[:, None, :], n_frames, axis=1)
        # frame k is (n_frames - 1 - k) steps in the past
        lag = np.arange(n_frames - 1, -1, -1, dtype=np.float32)[None, :]
        history[:, :, 0] -= lag * raw[:, None, 2]  # ball_x -= lag * vx
        history[:, :, 1] -= lag * raw[:, None, 3]  # ball_y -= lag * vy
        np.clip(history[:, :, 0], 0.0, GAME_W, out=history[:, :, 0])
        np.clip(history[:, :, 1], 0.0, GAME_H, out=history[:, :, 1])

        stacked = (history / _STATE_SCALE - _STATE_SHIFT) * 2.0
        out = [self._forward(stacked[i : i + batch_size]) for i in range(0, len(stacked), batch_size)]
        return np.concatenate(out, axis=0) if out else np.zeros((0, N_ACTIONS), dtype=np.float32)
