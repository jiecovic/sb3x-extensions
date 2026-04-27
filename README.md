# sb3x-extensions

`sb3x-extensions` is my unofficial extension package for
Stable-Baselines3 and `sb3-contrib`.

The Python package name is `sb3x`.

## Included Variants

The package currently focuses on exploration, hybrid, masked, and recurrent
variants around Stable-Baselines3 DQN/PPO/SAC, `sb3-contrib`'s `MaskablePPO`,
and `sb3-contrib`'s `RecurrentPPO`.

| Variant | Main idea |
| --- | --- |
| `BoltzmannDQN` | DQN with softmax-over-Q exploration instead of epsilon-greedy action selection. |
| `DiscreteSAC` | SAC for finite `Discrete` action spaces with exact action expectations. |
| `MaskableRecurrentPPO` | Recurrent PPO with invalid-action masks. |
| `HybridActionPPO` | PPO for `Dict(continuous=Box, discrete=MultiDiscrete)` action spaces. |
| `MaskableHybridActionPPO` | Hybrid-action PPO with masks applied only to the `MultiDiscrete` branch. |
| `HybridRecurrentPPO` | Recurrent PPO for the same hybrid action setup. |
| `MaskableHybridRecurrentPPO` | Hybrid-action recurrent PPO with masks applied only to the `MultiDiscrete` branch. |
| `HybridActionSAC` | SAC for hybrid actions, with an exact discrete-branch expectation and a Gaussian continuous branch. |
| `MaskableHybridActionSAC` | Hybrid-action SAC with masks applied only to the `MultiDiscrete` branch. |

## Status

This repository is experimental.

There is no guarantee that the implementation is fully correct, complete, or
fit for your use case. Use it at your own risk and validate it yourself before
depending on it for real work.

## Requirements

- Python 3.10 or newer
- `stable-baselines3` 2.8.0 or newer, below 3.0
- `sb3-contrib` 2.8.0 or newer, below 3.0

Those runtime dependencies are installed automatically when you install
`sb3x`.

## Installation

From a local clone:

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from sb3x import (
    BoltzmannDQN,
    DiscreteSAC,
    HybridActionPPO,
    HybridRecurrentPPO,
    HybridActionSAC,
    MaskableHybridActionPPO,
    MaskableHybridActionSAC,
    MaskableHybridRecurrentPPO,
    MaskableRecurrentPPO,
)
```

The intended API style is close to `sb3-contrib`'s `MaskablePPO` and
`RecurrentPPO`.

`BoltzmannDQN` expects the same `Discrete` action spaces as SB3's `DQN`.
The DQN loss, replay buffer, target network, and greedy target backup remain
unchanged; only non-deterministic action selection samples from
`softmax(Q(s, a) / temperature)`.

`DiscreteSAC` also expects `Discrete` action spaces. It uses a categorical
actor and twin Q-critics, computing SAC actor and target expectations exactly
over the finite action set.

`HybridActionPPO` expects an environment action space shaped like:

```python
spaces.Dict(
    {
        "continuous": spaces.Box(...),
        "discrete": spaces.MultiDiscrete(...),
    }
)
```

`MaskableHybridActionPPO` uses the same action space and expects
`env.action_masks()` to return the flattened `MultiDiscrete` mask, matching the
mask convention used by `sb3-contrib`'s `MaskablePPO`.

`HybridRecurrentPPO` uses the same hybrid action space and follows
`sb3-contrib`'s recurrent policy API for passing recurrent state through
`predict()`.

`MaskableHybridRecurrentPPO` combines both constraints: recurrent state follows
the same recurrent API, while `env.action_masks()` only masks the discrete
branch.

`HybridActionSAC` uses the same hybrid action space. The discrete branch is
enumerated exactly during SAC updates, so very large `MultiDiscrete`
combinations are intentionally rejected by default.

`MaskableHybridActionSAC` uses the same hybrid action space and applies
`env.action_masks()` only to the discrete branch. The SAC target and actor loss
integrate over the valid discrete actions for each sampled transition.

## Algorithm References

The implementations in this package are loosely based on the algorithmic ideas
in the references below. This is not a claim that `sb3x` exactly reproduces
every paper.

### Peer-Reviewed References

- DQN: Mnih et al., "Human-level control through deep reinforcement learning",
  Nature 518, 529-533, 2015. DOI:
  [10.1038/nature14236](https://doi.org/10.1038/nature14236)
- Boltzmann exploration: Cesa-Bianchi et al., "Boltzmann Exploration Done
  Right", Advances in Neural Information Processing Systems 30, 2017. Stable
  proceedings URL:
  <https://papers.nips.cc/paper/7208-boltzmann-exploration-done-right>
- SAC: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
  Reinforcement Learning with a Stochastic Actor", Proceedings of the 35th
  International Conference on Machine Learning, PMLR 80:1861-1870, 2018.
  Stable proceedings URL:
  <https://proceedings.mlr.press/v80/haarnoja18b.html>
- Invalid-action masking: Huang and Ontañón, "A Closer Look at Invalid Action
  Masking in Policy Gradient Algorithms", The International FLAIRS Conference
  Proceedings 35, 2022. DOI:
  [10.32473/flairs.v35i.130584](https://doi.org/10.32473/flairs.v35i.130584)
- Partial observability and recurrent RL motivation: Hausknecht and Stone,
  "Deep Recurrent Q-Learning for Partially Observable MDPs", AAAI Fall
  Symposium on Sequential Decision Making for Intelligent Agents, 2015. Stable
  author page:
  <https://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-SDMIA15-Hausknecht.html>
- Hybrid / parameterized action spaces: Masson et al., "Reinforcement Learning
  with Parameterized Actions", Proceedings of the Thirtieth AAAI Conference on
  Artificial Intelligence, 1934-1940, 2016. DOI:
  [10.1609/aaai.v30i1.10226](https://doi.org/10.1609/aaai.v30i1.10226)
- Hybrid actor-critic methods: Fan et al., "Hybrid Actor-Critic Reinforcement
  Learning in Parameterized Action Space", Proceedings of the Twenty-Eighth
  International Joint Conference on Artificial Intelligence, 2279-2285, 2019.
  DOI: [10.24963/ijcai.2019/316](https://doi.org/10.24963/ijcai.2019/316)

### Canonical Preprints

These are useful implementation references, but arXiv is not peer review.

- PPO: Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
  arXiv: <https://arxiv.org/abs/1707.06347>
- Automatic entropy tuning in SAC: Haarnoja et al., "Soft Actor-Critic
  Algorithms and Applications", 2018. arXiv:
  <https://arxiv.org/abs/1812.05905>
- Discrete SAC: Christodoulou, "Soft Actor-Critic for Discrete Action
  Settings", 2019. arXiv: <https://arxiv.org/abs/1910.07207>

## Related Projects

- Stable-Baselines3 docs: <https://stable-baselines3.readthedocs.io/>
- Stable-Baselines3 GitHub: <https://github.com/DLR-RM/stable-baselines3>
- sb3-contrib docs: <https://sb3-contrib.readthedocs.io/>
- sb3-contrib GitHub: <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib>

`sb3x` is unofficial and not affiliated with the Stable-Baselines3 or
`sb3-contrib` maintainers.

## Notes

Repo-local training, watch, and MiniGrid validation tooling lives under
`tools/` and `scripts/`. That support code is for development and testing in
this repository; it is not the installed library surface.

## License

This project is MIT licensed. See [LICENSE](LICENSE).
