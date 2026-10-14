.PHONY: install test lint docs build clean

install:
	uv sync --extra cuda12

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check

docs:
	uv run --group docs mkdocs build --strict

build:
	uv build

clean:
	rm -rf build dist *.egg-info
