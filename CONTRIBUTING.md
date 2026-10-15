# Contributing

Thanks for your interest in the Rius Python SDK (`glassflow-rius` on PyPI,
imported as `rius`). This guide covers how to set up, test and submit a change.

To report a security problem, follow [SECURITY.md](SECURITY.md) instead of
opening an issue.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) and supports Python 3.10 and
newer.

```bash
uv sync --group dev
```

Commit `uv.lock` whenever dependencies change. Use `uv` and `ruff` only: no
pip/venv workflows and no other formatter or linter.

## Checks

All of these must pass before you open a pull request, and CI runs them on
every pull request (tests on Python 3.10 to 3.14):

```bash
uv run ruff check .          # lint
uv run ruff format --check . # format (run without --check to fix)
uv run mypy                  # type-check (strict)
uv run pytest                # tests
```

## Code conventions

- Fully typed with modern syntax (`X | None`, built-in generics); `mypy --strict`
  must pass.
- Ruff handles both linting and formatting, with a line length of 100.
- Public API is re-exported from `src/rius/__init__.py` and listed in `__all__`.
- Prefer dependency injection over mocking (for example
  `init(span_exporter=...)` instead of patching the OTLP exporter).
- Attribute keys come from `rius.semconv`. The generated module
  `src/rius/_semconv_generated.py` is vendored from
  [rius-semconv](https://github.com/glassflow/rius-semconv); never edit it by
  hand. Change the registry there, then re-pin with
  `TAG=<tag> scripts/sync-semconv.sh`.

## Tests first

We work test-first: write a failing test, watch it fail for the right reason,
then write the minimal code that makes it pass. Every behaviour change or bug
fix should come with a test that would have caught it.

## Pull requests

- Branch from `main`.
- Pull requests are squash-merged, and the PR title becomes the commit message
  on `main`. [release-please](https://github.com/googleapis/release-please)
  reads it to decide the next version, so the title must be a
  [Conventional Commit](https://www.conventionalcommits.org/):
  - `feat:` for a new feature (minor release)
  - `fix:` for a bug fix (patch release)
  - `feat!:` (or `fix!:`) for a breaking change; mark it in the title, since a
    `BREAKING CHANGE:` footer in a branch commit does not survive the squash
  - `docs:`, `test:`, `refactor:`, `chore:`, `ci:` for changes that do not
    release on their own
- Do not edit the version or `CHANGELOG.md`; release-please maintains both.
- A Developer Certificate of Origin sign-off is not required.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
