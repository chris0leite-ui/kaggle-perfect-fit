# Kaggle / ML Competition Playbook

Portable scaffold for Kaggle-style ML competitions. Clone or copy into
a new competition repo at kickoff to get a consistent `CLAUDE.md`
(loaded as memory every session), a kickoff skill, and a place to drop
reusable scripts.

Derived from "The Perfect Fit" competition (Apr 2026).

## Files

- **`CLAUDE.md`** — playbook instructions. Loaded as memory by Claude
  Code every session. Rename / merge into the new competition repo's
  `CLAUDE.md`.
- **`.claude/skills/kaggle-kickoff/SKILL.md`** — invokable via
  `/kaggle-kickoff`. Walks Day 1 setup (brief.md, LB budget, rank,
  deadline, CLI credential check, daily log bootstrap).

## How to bootstrap a new competition

From a fresh competition repo:

```bash
# clone this scaffold into a scratch dir
git clone -b claude/kaggle-playbook <this-repo-url> /tmp/kaggle-playbook

# copy the playbook into the new repo
cp /tmp/kaggle-playbook/CLAUDE.md ./CLAUDE.md
mkdir -p .claude/skills
cp -r /tmp/kaggle-playbook/.claude/skills/kaggle-kickoff .claude/skills/

# commit to the new repo
git add CLAUDE.md .claude
git commit -m "Add Kaggle competition playbook scaffold"
```

Then in the first session, invoke `/kaggle-kickoff` and work through
Day 1.

## Maintaining the scaffold

Treat this branch as the canonical playbook. When a competition
surfaces a new transferable lesson, update `CLAUDE.md` here and
cherry-pick into active competition repos as you go.
