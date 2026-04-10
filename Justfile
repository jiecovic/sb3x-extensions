set positional-arguments := true

default:
    @just --list

install-dev:
    uv sync --python 3.11 --extra dev

format:
    uv run --python 3.11 --extra dev ruff format .

format-check:
    uv run --python 3.11 --extra dev ruff format --check .

lint:
    uv run --python 3.11 --extra dev ruff check .

typecheck:
    uv run --python 3.11 --extra dev pyright

test:
    uv run --python 3.11 --extra dev pytest

test-minigrid:
    uv run --python 3.11 --extra dev pytest tests/test_minigrid_recurrent_ppo.py

train *args:
    uv run --python 3.11 --extra dev python -m tools.minigrid_memory.train {{args}}

watch *args:
    uv run --python 3.11 --extra dev python -m tools.minigrid_memory.watch {{args}}

build:
    uv run --python 3.11 --extra dev python -m build

check: format-check lint typecheck test

ci: check build

minigrid-compare *args:
    uv run --python 3.11 --extra dev python -m scripts.compare_minigrid_memory {{args}}
