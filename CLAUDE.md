# Kaggle / ML Competition Kickoff Playbook

Load at the start of any new competition (new repo, new session). Ask
for these proactively; do not wait to be prompted.

## Day 1

1. **`brief.md` in the repo**: paste the full host material —
   description, rules, eval page, data description, host forum/notebook
   comments. Invariances and constraints often live here.
2. **LB submission budget**: daily limit, total limit, used so far.
   Track remaining count; rank candidates by expected information gain.
3. **Current LB rank + distances** to the clusters the user cares about
   (top-N, median, target).
4. **Deadline + weekly hours** the user expects to put in.
5. **Between-session channels**: forum, CSV inspection, collaborators.
   Surface anything the user sees that I don't.
6. **Tooling check**: `which kaggle`, `ls ~/.kaggle/kaggle.json`, and
   `env | grep -i kaggle`. If missing on Claude Code web, ask the user
   to set `KAGGLE_USERNAME` and `KAGGLE_KEY` in the Environment
   settings UI (Settings → Environment variables) — those persist
   across cloud sessions; chat paste is a fallback only. If `kaggle`
   CLI is absent, `pip install --user kaggle`.

## Workflow across sessions

- **Daily log** (bounded, one paragraph) at the top of CLAUDE.md: goal,
  what changed, LB delta, next bet. Narrative log stays underneath.
- **Hypothesis board**: open / ruled-out / parked. Keep it explicit.
- **Stop-conditions per branch**: agree a rule up front (e.g. "if first
  variant is within X CV of baseline, kill the branch").
- **Prune outputs**: ask which plots/reports the user actually uses.
- **Narrate "smells off" triggers**: when the user flags something I
  didn't propose, ask why, so I can probe similar angles unprompted.

## Methodology

- Understand the problem before optimising. Interpretable models are
  one route; causal discovery, pooled-feature shift analysis, seed
  recovery, and reading host material are others.
- CV–LB divergence is a diagnostic. The multiplier measures how much
  training-specific structure is being exploited.
- DGP archaeology is a distinct phase with its own tools.
- Record dead ends alongside wins.
- Transferable method > reproducible result.
- Gaps between model families carry signal (linear vs GAM =
  nonlinearity; GAM vs EBM = interactions).
- Human + AI roles are complementary and evolve; do not fix them.

## TDD / scaffolding

Red-green TDD against `src/` + `tests/` helps during modelling (clean
interfaces, preprocessor correctness). It becomes friction during
archaeology, where one-off scripts are faster. Plan to archive to
`legacy/` once the pipeline stabilises.
