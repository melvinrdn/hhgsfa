# Development

The project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/melvinrdn/hhgsfa
cd hhgsfa
uv sync --group dev --extra numba
uv run pytest
```

Lint and format with [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`.

```bash
uvx ruff check .
uvx ruff format .
```

Build the documentation, with warnings as errors.

```bash
uv run --group docs sphinx-build -W docs docs/_build/html
```

The benchmark against StrongFieldApproximation.jl needs Julia, see [its page](benchmark.md).
