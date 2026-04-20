---
name: kaggle-kickoff
description: Use at the start of a new Kaggle/ML competition to walk through Day 1 setup — brief.md collection, LB submission budget, current rank, deadline, between-session channels, tooling and credential check, daily log bootstrap. Invoke on the very first session in a new competition repo.
---

# Kaggle kickoff

Goal: by the end of this walk-through, the competition repo has a
`brief.md`, an up-to-date `CLAUDE.md` daily log section, a known LB
submission budget and rank, and a working `kaggle` CLI with credentials.

Work through each step in order. Ask the user one question per step
(or a small group) and wait for the answer before moving on. Do not
batch.

## Step 1 — brief.md

Ask the user to paste:

- competition description
- rules page
- evaluation metric details
- data description
- any host forum/notebook comments they consider relevant

Save verbatim as `brief.md` in the repo root. Before moving on, scan
it for **invariances and constraints** (e.g. "test must be in the
convex hull of training", "metric is MAE not RMSE", "only tabular
features allowed"). Note anything that could constrain the DGP or the
modelling approach.

## Step 2 — LB submission budget

Ask:

- daily submission limit
- total submission limit
- submissions already spent (0 if fresh)

Record in `CLAUDE.md` under a "Submission budget" heading. Going
forward, rank candidate submissions by **expected information gain**,
not CV score alone — a submission that discriminates between two
hypotheses is worth more than one that re-confirms a leader.

## Step 3 — current LB state

Ask:

- current rank (or "not yet submitted")
- score distance to top-N, median, and the user's target

This shapes the strategy: a 5-point gap to target suggests incremental
tuning; a 30-point gap suggests DGP archaeology or a qualitative model
switch.

## Step 4 — deadline + cadence

Ask:

- competition close date
- weekly hours the user expects to put in

Budgets depth of long-shot investigations (seed recovery, causal
discovery, etc.).

## Step 5 — between-session channels

Ask what the user does outside Claude Code sessions:

- reading the Kaggle discussion forum
- manual CSV inspection
- collaborators
- other ML tools / notebooks

Surface anything they see that the session won't. If the user is
pulling signal from forum posts, ask them to paste key findings into
`CLAUDE.md` so it reaches future sessions.

## Step 6 — tooling and credential check

Run:

```bash
which kaggle
ls ~/.kaggle/kaggle.json 2>/dev/null
env | grep -i kaggle
```

Interpret:

- If `kaggle` CLI is missing: `pip install --user kaggle`.
- If credentials are missing AND this is Claude Code web:
  - Tell the user to open Settings → Environment variables in the
    Claude Code web UI and add `KAGGLE_USERNAME` and `KAGGLE_KEY`.
    That UI persists across cloud sessions; chat paste is a fallback
    only since the token would land in transcript logs.
- If credentials are missing AND this is a local install: guide the
  user to drop their token at `~/.kaggle/kaggle.json` and
  `chmod 600`.

Verify with `kaggle competitions list -s <slug>` or similar.

## Step 7 — initialise the daily log

Add to `CLAUDE.md` near the top:

```
## Session log

### YYYY-MM-DD — kickoff

- Goal: ...
- Changed: ...
- LB delta: ...
- Next bet: ...
```

Fill in today's entry with what we just accomplished in this kickoff.
Keep each future session entry to one paragraph; the full narrative
log stays underneath.

## Step 8 — hypothesis board

Add a lightweight board to `CLAUDE.md`:

```
## Hypothesis board

- **Open**: ...
- **Ruled out**: ...
- **Parked**: ...
```

Seed it from anything `brief.md` or Step 1 surfaced. Update as the
competition progresses.

## Handoff

Summarise in 3–5 bullets:

- what we learned in this kickoff
- what's in `brief.md` worth flagging
- the LB budget and rank state
- any blockers (missing credentials, missing data, etc.)
- the proposed first experiment

Ask the user to confirm before any modelling work begins.
