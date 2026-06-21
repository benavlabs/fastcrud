# Library Skill for AI Coding Agents

FastCRUD ships with a bundled [Library Skill](https://github.com/tiangolo/library-skills) inside the installed package at `fastcrud/.agents/skills/fastcrud/SKILL.md`. Library Skills follow the [Agent Skills](https://agentskills.io/) standard, so any compatible AI coding tool can use it — Codex, Cursor, GitHub Copilot, OpenCode, Claude Code, Pi, Antigravity, and [30+ others](https://agentskills.io/clients).

## Quick install

After `fastcrud` is installed in your project (`uv add fastcrud` / `pip install fastcrud`), run [`library-skills`](https://github.com/tiangolo/library-skills) from the project root:

=== "Python (uvx)"

    ```sh
    uvx library-skills              # for Codex, Cursor, Copilot, OpenCode, etc.
    uvx library-skills --claude     # for Claude Code (uses .claude/skills)
    ```

=== "Node.js (npx)"

    ```sh
    npx library-skills              # for Codex, Cursor, Copilot, OpenCode, etc.
    npx library-skills --claude     # for Claude Code (uses .claude/skills)
    ```

This symlinks the bundled skill into your project's `.agents/skills/` directory (or `.claude/skills/` with `--claude`). Because they're symlinks, **upgrading `fastcrud` automatically updates the skill** — no need to re-run.

Re-run `library-skills` to pick up new or moved skills after upgrades, or to clean up broken symlinks. See the [Library Skills `Use` docs](https://github.com/tiangolo/library-skills) for `--check`, `--yes`, and `--copy` (for systems without symlink support, like some Windows setups).

## What the skill teaches agents

- **Canonical setup** — the minimal model + schemas + `crud_router` pattern
- **Choosing the right method** — `get` vs `get_joined` vs `get_multi_joined`, and which one avoids N+1
- **Avoiding N+1** — auto-detection via `include_relationships=True`, manual `JoinConfig`, and `nested_limit` for one-to-many collections
- **The `limit=None` footgun** — when it's safe (domain-bounded WHERE) and when it isn't, with the three-tier pagination default
- **When NOT to use FastCRUD** — aggregate roll-ups, CTEs, `GROUP BY` projections, bulk writes, and dialect-specific features that belong to raw SQLAlchemy
- **Return-value semantics** — `create()` returns `None` by default, `return_as_model=True` requires `schema_to_select`, type narrowing via subclass overrides
- **Filter syntax** — every `__op` suffix, joined filters with `.`, `Depends(...)` filters, custom operators
- **Gotchas** — async-session-only, soft delete behavior, joined-table inheritance, SQLModel polymorphism caveat, Python 3.14 PEP 649 compatibility

## Layout

```
fastcrud/.agents/skills/fastcrud/
├── SKILL.md              # entry point — loaded when the agent activates the skill
└── references/
    ├── methods.md        # every FastCRUD method, return semantics, type narrowing
    ├── filters.md        # every operator, FilterConfig, dependency filters, custom operators
    ├── joins.md          # JoinConfig, auto-detection, nested_limit, polymorphism
    ├── pagination.md     # offset vs cursor, sanity-checking limit=None
    └── endpoints.md      # crud_router params, EndpointCreator subclassing, soft delete
```

The agent loads only the skill's name and short description at startup. When a task matches, the full `SKILL.md` body loads into context. Reference files load only on demand — this pattern is called [progressive disclosure](https://agentskills.io/specification#progressive-disclosure).

## Verifying it's loaded

In Claude Code, type `/skills` after installing — the `fastcrud` skill should appear in the list. In other clients, consult their documentation for how to view available skills.

## Contributing improvements

The skill ships from the FastCRUD repository at [`fastcrud/.agents/skills/fastcrud/`](https://github.com/benavlabs/fastcrud/tree/main/fastcrud/.agents/skills/fastcrud). If you notice incorrect behavior, missing gotchas, or anti-patterns the skill should flag, please open an issue or PR with a concrete example.
