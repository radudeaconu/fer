# Changes

Rolling log of every code modification. Format: `- <file>: <what> — <why>` under a daily heading.
See `CLAUDE.md` Rule 1 for the contract.

## 2026-04-30

- `.gitignore`: Initial ignore rules — exclude data/, checkpoints/, runs/, third_party/, Python bytecode, notebook checkpoints, and `.claude/` (local Claude tooling state). Keeps the repo focused on source code, not artifacts or per-user tooling.
- `CLAUDE.md`: Authored the project rules file — locks in datasets/models, enforces CHANGES.md + per-change git commits, marks research/ read-only. Required so future sessions stay consistent with the approved plan.
- `CHANGES.md`: Created the change log with seed entries for the bootstrap files. Establishes the documentation discipline from the first commit.
