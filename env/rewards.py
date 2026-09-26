"""Reward specification for the DECOY environment.

The environment previously exposed no reward signal at all: ``Agent.reward``
returned a field that was initialised to zero and never written. This module
supplies a configurable default so the environment can actually be trained
against, while keeping the shaping terms explicit and easy to replace.

Nothing here is claimed to be the "correct" reward for competitive CS:GO -- it
is a reasonable, documented starting point. Every term can be re-weighted, and
setting all shaping terms to zero recovers the pure win/loss objective.

Sign convention: rewards are from the perspective of the agent receiving them,
so ``damage_taken`` and ``death`` are negative.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class RewardConfig:
    """Weights for each reward event.

    Attributes:
        win / lose: terminal reward granted to every member of the
            winning / losing team when the round resolves.
        kill / death: credited to the attacker and the victim respectively at
            the moment an agent's health reaches zero.
        damage_dealt / damage_taken: dense shaping, applied per point of HP so
            a full 100 HP elimination is worth ``100 * damage_dealt`` on top of
            the ``kill`` bonus.
        bomb_plant / bomb_defuse: credited to the agent performing the action.
            The rest of that agent's team receives it scaled by ``team_spirit``.
        time_penalty: small negative applied once per decision request, which
            discourages stalling without dominating the objective.
        objective_progress: potential-based shaping, per unit of graph distance
            closed on the current objective (a bomb site before the plant, the
            planted bomb afterwards). Because it is the difference of a
            potential over states, it does not change the set of optimal
            policies (Ng, Harada & Russell, 1999) -- it only makes the gradient
            findable. Defaults to 0; without it a random policy never reaches a
            bomb site, so win/loss carries no signal at all and nothing learns.
        team_spirit: in ``[0, 1]``. Fraction of an individual's event reward
            that is also handed to each teammate. ``0`` is fully selfish credit
            assignment, ``1`` makes the team's reward fully shared. Terminal
            win/lose rewards are always team-wide and ignore this.
    """

    win: float = 1.0
    lose: float = -1.0
    kill: float = 0.5
    death: float = -0.5
    damage_dealt: float = 0.01
    damage_taken: float = -0.005
    bomb_plant: float = 0.3
    bomb_defuse: float = 0.3
    time_penalty: float = -0.001
    objective_progress: float = 0.0
    team_spirit: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.team_spirit <= 1.0:
            raise ValueError(f"team_spirit must be in [0, 1], got {self.team_spirit}")

    def to_dict(self) -> dict:
        return asdict(self)


#: Dense default used when the environment is constructed without an explicit config.
DEFAULT_REWARD_CONFIG = RewardConfig()

#: Dense default plus objective-distance shaping. This is what the bundled MARL
#: baseline trains against: with the unshaped default a random policy never
#: reaches a bomb site, so the terminal reward is constant and nothing learns.
SHAPED_REWARD_CONFIG = RewardConfig(objective_progress=0.01)

#: Pure outcome reward: no shaping, only the terminal win/loss signal. Harder to
#: learn from, but free of the bias that shaping terms introduce.
SPARSE_REWARD_CONFIG = RewardConfig(
    kill=0.0,
    death=0.0,
    damage_dealt=0.0,
    damage_taken=0.0,
    bomb_plant=0.0,
    bomb_defuse=0.0,
    time_penalty=0.0,
)
