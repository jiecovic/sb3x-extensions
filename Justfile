set positional-arguments := true

python := env_var_or_default("PYTHON", "python")
ruff := env_var_or_default("RUFF", "ruff")
pyright := env_var_or_default("PYRIGHT", "pyright")

default:
    @just --list

install-dev:
    {{python}} -m pip install -e ".[dev]"

format:
    {{ruff}} format .

format-check:
    {{ruff}} format --check .

lint:
    {{ruff}} check .

typecheck:
    {{pyright}}

test:
    {{python}} -m pytest

test-minigrid:
    {{python}} -m pytest tests/test_minigrid_recurrent_ppo.py

train *args:
    {{python}} -m tools.minigrid_memory.train {{args}}

watch *args:
    {{python}} -m tools.minigrid_memory.watch {{args}}

build:
    {{python}} -m build

check: format-check lint typecheck test

ci: check build

minigrid-compare *args:
    {{python}} -m scripts.compare_minigrid_memory {{args}}
