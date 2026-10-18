# Contributing to PIR

PIR is maintained by one person (Qazi M. Hanif). Contributions, bug reports
and questions are welcome.

- **Bugs and questions:** open a GitHub issue. Please include a small data
  example (or code that generates one), the PIR version
  (`python -c "import physics_engine; print(physics_engine.__version__)"`),
  and the full error message.
- **Pull requests:** keep them focused; add or update a test in `tests/` for
  any behaviour change; run `python -m pytest -q tests` before opening the PR.
  CI runs the same tests on every push and pull request.
- **Benchmark claims:** results in the README must come from runs that can be
  reproduced; please state the dataset, seeds, noise level and scorer used.

Response time is best effort; the project has no paid support.
