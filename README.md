# sb3x-extensions

`sb3x-extensions` is my unofficial extension package for
Stable-Baselines3 and `sb3-contrib`.

The Python package name is `sb3x`.

It currently contains:

- `MaskableRecurrentPPO`: a local combination of the ideas behind
  `MaskablePPO` and `RecurrentPPO`.
- `HybridActionPPO`: an experimental PPO variant for hybrid
  `Dict(continuous=Box, discrete=MultiDiscrete)` action spaces.

## Status

This repository is experimental.

There is no guarantee that the implementation is fully correct, complete, or
fit for your use case. Use it at your own risk and validate it yourself before
depending on it for real work.

## Requirements

- Python 3.10 or newer
- `stable-baselines3`
- `sb3-contrib`

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
from sb3x import HybridActionPPO, MaskableRecurrentPPO
```

The intended API style is close to `sb3-contrib`'s `MaskablePPO` and
`RecurrentPPO`.

`HybridActionPPO` expects an environment action space shaped like:

```python
spaces.Dict(
    {
        "continuous": spaces.Box(...),
        "discrete": spaces.MultiDiscrete(...),
    }
)
```

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

This project is MIT licensed. See [LICENSE](/home/thomas/projects/sb3x-extensions/LICENSE).
