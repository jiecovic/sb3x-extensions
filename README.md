# sb3x-extensions

`sb3x-extensions` is an unofficial extension package for
Stable-Baselines3 and `sb3-contrib`.

The repository name stays distinct from the upstream projects, while the
Python package name is `sb3x`.

## Status

This repository now contains a local `MaskableRecurrentPPO` package scaffold
based on `sb3-contrib`'s recurrent PPO implementation.

At the moment, that class is an explicit starting point for follow-up work. It
does not add action-masking support yet, so its behavior still matches
recurrent PPO.

## Goals

- Keep a standalone home for SB3-adjacent extensions.
- Make future extensions installable and testable as a normal Python package.
- Stay explicit about being unofficial and unaffiliated with the
  Stable-Baselines3 maintainers.

## Planned First Extension

The first concrete target is `MaskableRecurrentPPO`.

The current implementation strategy is:

- keep a local fork of the recurrent PPO surface under the target name
  `MaskableRecurrentPPO`
- integrate maskable behavior incrementally on top of that local code

The follow-up work will likely require a few pieces to land together:

- a recurrent rollout buffer that carries action masks
- masked action distribution support for recurrent `MultiDiscrete` policies
- recurrent inference and `predict()` handling that accepts masks alongside
  recurrent state

Those pieces are better introduced as a coherent implementation than as empty
stubs.

## Package Layout

The current source tree stays close to the `sb3-contrib` algorithm-package
layout:

```text
src/
  sb3x/
    __init__.py
    py.typed
    maskable_recurrent/
      __init__.py
      maskable_recurrent_ppo.py
      policies.py
scripts/
  compare_minigrid_memory.py
tools/
  minigrid_memory/
    models.py
    support.py
    runs.py
    train.py
    watch.py
Justfile
tests/
  test_imports.py
  test_minigrid_runs.py
  test_minigrid_recurrent_ppo.py
```

This keeps the first algorithm self-contained. If real shared code emerges
later, it can be split out then instead of scaffolding empty `common/`
packages up front.

## Installation

`sb3x` targets Python 3.11 and newer.

Install the package from a local clone with:

```bash
pip install .
```

The intended public import for the algorithm is:

```python
from sb3x import MaskableRecurrentPPO
```

The `sb3x.maskable_recurrent` package remains available as the algorithm
namespace, but users should not need it for the main class import.

For development:

```bash
pip install -e ".[dev]"
```

## Development Checks

After installing the development extras, the main local checks are:

```bash
ruff check .
ruff format --check .
pyright
pytest
```

If you use `just`, the repo also includes a small [Justfile](/home/thomas/projects/sb3x-extensions/Justfile) with convenience commands:

```bash
just check
just build
just train --algo local --timesteps 100000
just watch --run-dir local/runs/<run-name>
just minigrid-compare --train-timesteps 256
```

## MiniGrid Validation

The MiniGrid train/watch/validation workflow is intentionally repo-local under
`tools/` and `scripts/`. It is not part of the installed `sb3x` package.

The repo now uses two separate MiniGrid paths:

- parity path:
  `FlatObsWrapper` + `MlpLstmPolicy`
- benchmark path:
  `ImgObsWrapper` + custom `MinigridFeaturesExtractor` + `CnnLstmPolicy`

The parity path exists only to make sure the local recurrent PPO copy still
matches upstream before maskable changes land.

The benchmark path is the one used by `train` and `watch`. It keeps the
MiniGrid image structure intact instead of flattening the symbolic grid and
mission text into one long vector.

There are two intended checks:

- make sure upstream `sb3-contrib` `RecurrentPPO` runs cleanly on the env
- make sure the local `MaskableRecurrentPPO` copy matches upstream before any
  masking changes land

The automated checks live in [tests/test_minigrid_recurrent_ppo.py](/home/thomas/projects/sb3x-extensions/tests/test_minigrid_recurrent_ppo.py).

For a manual comparison run:

```bash
python -m scripts.compare_minigrid_memory --train-timesteps 256
```

That script prints:

- a deterministic step-by-step `predict()` parity trace
- post-train evaluation summaries for upstream and local models
- the maximum policy parameter difference after both runs

It intentionally keeps using the flat parity setup. It is not the benchmark
training path.

## Train And Watch

There is now a minimal local train/watch workflow for `MiniGrid-MemoryS7-v0`.

Train either upstream recurrent PPO or the local copied algorithm into a saved
run directory:

```bash
just train --algo upstream --timesteps 100000
just train --algo local --timesteps 100000
```

Those commands now default to the image benchmark setup:

- observation mode: `image`
- policy: `CnnLstmPolicy`
- feature extractor: repo-local `MinigridFeaturesExtractor`

The train CLI now prints a structured preflight summary of:

- the saved run configuration
- the chosen policy hyperparameters
- the instantiated torch policy modules

Useful switches:

```bash
just train --algo local --timesteps 100000 --progress-bar
just train --algo local --timesteps 100000 --verbose 1
just train --algo local --timesteps 100000 --no-tensorboard
just train --algo local --observation-mode flat
```

Each run writes a directory under `local/runs/` containing:

- `run_config.json`
- `final_model.zip`
- `checkpoints/`
- `eval/`
- `tensorboard/`

If tensorboard logging is enabled, you can inspect it with:

```bash
tensorboard --logdir local/runs
```

To watch a saved run:

```bash
just watch --run-dir local/runs/<run-name>
```

By default, watch mode now follows `latest` and keeps running until you stop it.
It checks the watched artifact between episodes and reloads it when a newer
checkpoint appears on disk.

For a headless smoke run, use:

```bash
just watch --run-dir local/runs/<run-name> --artifact final --render-mode rgb_array --episodes 1
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
