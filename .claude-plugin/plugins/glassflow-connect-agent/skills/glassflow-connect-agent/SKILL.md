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

## 3. Manual dry run (for whoever implements this task, not part of the shipped skill)

Before moving to Task 4, manually walk through steps 1-2 above against a real workspace once as an admin and once as a member, using the `create_api_key` tool built in Task 1 (point the MCP server at a local/dev argus-core, or use `respx`-style manual curl calls to confirm the response text matches what's checked above verbatim). Confirm the exact wording emitted by `create_api_key` in Task 1's Step 3 implementation is what the branches above key off of — update either side if they've drifted.
