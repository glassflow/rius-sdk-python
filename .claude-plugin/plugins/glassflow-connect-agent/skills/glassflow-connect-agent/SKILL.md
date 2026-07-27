---
name: glassflow-connect-agent
description: Connect this repo's agent to GlassFlow observability — mints an API key, installs the SDK, wires it into your code, runs your agent, and shows you your first trace in the UI.
---

# Connect this agent to GlassFlow

Run this procedure top to bottom. Do not skip the admin check before attempting to create a key — a non-admin call is a wasted round trip you can avoid by checking first.

## 1. Check for an existing key

Look for `GLASSFLOW_API_KEY` in this repo's `.env` file (or already set in the shell environment). If it's already there, tell the user you found an existing key and are skipping key creation, then go straight to "Install the SDK".

## 2. Check the caller's role

Call the `get_me` MCP tool. Read its `**Organization**` section:

- If it says `none (you have not joined an organization yet)` — **stop here**. Tell the user: "You don't belong to a workspace yet — if someone invited you, accept that invite link first, then re-run this skill." Do not proceed to any key step.
- If the role shown is `admin` — proceed to 2a.
- If the role shown is `member` (or anything other than `admin`) — proceed to 2b.

### 2a. Admin: mint a key

Call the `create_api_key` MCP tool with no arguments (uses the caller's default workspace). Its response either starts with `"Created a new API key in workspace '"` (success — extract the plaintext key from the text after `"Save it now — it will not be shown again:"`, which appears on its own line following a blank line) or with `"You need admin access"` / contains `"Workspace not found"` (treat these as stop conditions and relay the message verbatim; the role check above means the admin-message case should be rare, but the tool can still return it, e.g. if role changed between the check and the call).

State to the user, plainly: `Created a new API key in workspace '<name>' and I'm about to save it to .env as GLASSFLOW_API_KEY.`

### 2b. Non-admin: ask for a key

Tell the user: `You're a member (not an admin) in workspace '<name>', so I can't mint a key myself — paste an existing API key below (ask a workspace admin if you don't have one), or press Enter to skip and add it yourself later.`

Wait for their reply.
- If they paste a key: state `Saving the key you gave me to .env as GLASSFLOW_API_KEY.`
- If they skip: tell them clearly that `.env` will not get a `GLASSFLOW_API_KEY` line, and that the rest of this skill (SDK install, entrypoint patch) will still run, but the run/verify steps later will not produce a trace until a real key is added. Do not fabricate a placeholder key.

## 3. Install the SDK

Detect the project's package manager by which file exists at the repo root, in this order: `uv.lock` → `uv add glassflow-ai`; `pyproject.toml` (no `uv.lock`) → `pip install glassflow-ai` inside the project's existing venv if one is active, otherwise `pip install glassflow-ai`; `Pipfile` → `pipenv install glassflow-ai`; `requirements.txt` or none of the above → `pip install glassflow-ai`. Run the chosen command and show its output.

## 4. Write `.env`

Determine `GLASSFLOW_SERVICE_NAME` as the repo's directory name, lowercased, with anything that isn't `[a-z0-9-]` replaced by `-`.

If `.env` exists, append (never overwrite existing lines):
```
GLASSFLOW_API_KEY=<key from step 2, if one was obtained>
GLASSFLOW_SERVICE_NAME=<computed name>
GLASSFLOW_HEARTBEAT=true
```
If `.env` doesn't exist, create it with just those lines. If no key was obtained (user skipped in step 2b), omit the `GLASSFLOW_API_KEY` line entirely and say so.

State the exact file path you wrote to, e.g. `Wrote /path/to/repo/.env`.

## 5. Find the entrypoint

Look for `AGENTS.md` or `CLAUDE.md` at the repo root first. If either exists, read it for an explicit description of the project's entrypoint or main script — use whatever it names.

If neither exists or neither names an entrypoint clearly, fall back to a heuristic scan of the repo root (not subdirectories) for `main.py` or `app.py`, in that order. If both exist, or neither exists, ask the user: "Which file is your agent's entrypoint?"

## 6. Patch the entrypoint

Near the top of the entrypoint file (after its existing imports), insert:
```python
import glassflow
glassflow.init(service_name="<GLASSFLOW_SERVICE_NAME from step 4>")
```

Then find the function that looks like the agent's main entry — the function called from `if __name__ == "__main__":`, or if there's no such guard, the sole top-level function that takes a string argument shaped like a query/prompt. Add `@observe` immediately above its `def` line (import `observe` from `glassflow` alongside `glassflow` itself: `from glassflow import observe`). Decorate exactly one function. If no single obvious candidate exists, ask the user which function to decorate rather than guessing.

Show the user a before/after diff of the entrypoint file.

## 7. Ask for a run command

Ask the user: "What command runs your agent or your eval set?" Accept any shell command (e.g. `python main.py`, `pytest tests/eval`).

## 8. Run it

Execute the given command with the repo's `.env` loaded into the environment. Stream its output back to the user as-is, including failures. If it exits non-zero, stop here — report the failure and do not proceed to verification, since there is nothing to verify.

## 9. Verify a trace arrived

Call the `list_agent_traces` MCP tool with `service=<GLASSFLOW_SERVICE_NAME>` and `hours=1`. If its response is `_No results._`, wait 5 seconds and try again, up to 6 total attempts (~30 seconds).

- If a trace shows up (the response is a markdown table with at least one row): proceed to handoff.
- If still empty after 6 attempts: tell the user a trace didn't arrive, and list the likely causes: network egress to the SDK's configured endpoint may be blocked, the command run in step 8 may not be the one that actually executes the patched entrypoint, or an exception may have prevented any span from closing (point them back at the step 8 output). Do not print a UI link in this case.

## 10. Hand off

Print:
- The UI link: `https://<ui-host>/w/<workspaceId>/traces?service=<GLASSFLOW_SERVICE_NAME>` (substitute the real UI host for this deployment and the `workspaceId` from the `get_me`/`create_api_key` call earlier in this run).
- These example prompts to try next, verbatim:
  - "Summarize my agent's traces from the last hour"
  - "Show me the slowest trace from that run"
  - "Did any of those spans error?"

---

## Manual dry run (for whoever implements this task, not part of the shipped skill — not one of the numbered steps above)

Before moving to Task 4, manually walk through steps 1-2 above against a real workspace once as an admin and once as a member, using the `create_api_key` tool built in Task 1 (point the MCP server at a local/dev argus-core, or use `respx`-style manual curl calls to confirm the response text matches what's checked above verbatim). Confirm the exact wording emitted by `create_api_key` in Task 1's Step 3 implementation is what the branches above key off of — update either side if they've drifted.
