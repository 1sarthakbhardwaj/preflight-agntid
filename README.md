# PreFlight — an AI creative director you can actually trust

**Nothing goes live until it has flown.** PreFlight is an AI creative director that takes a brief from Slack, generates ad creatives with ElarisLabs, test-launches every variant on a 100-agent synthetic social network, and sends the winner to a human for sign-off. **AgntID** enforces what the agent is allowed to do on every single tool call.

> "PreFlight predicts whether your post will go viral. AgntID makes sure the agent doesn't go rogue."

**Links:** [AgntID](https://agntid.ai) · [AgntID docs](https://docs.agntid.ai) · [ElarisLabs](https://elarislabs.ai) · [ElarisLabs Creative Studio](https://studio.elarislabs.ai) · [OASIS paper](https://arxiv.org/abs/2411.11581) · [AgntID Hackathon on Devfolio](https://agntid.devfolio.co) · [Inspiration: Eddy Xu's viral-prediction demo](https://x.com/eddybuild/status/1895968364046020608)

Built for the **AgntID Hackathon** (Bengaluru, 10 October 2026).

## The problem
Brands are handing AI agents three dangerous things:
1. **A credit card**: generation credits (images and video cost real money; a runaway request burns budget).
2. **A megaphone**: Slack and social accounts (unapproved drafts leaking to a client, or a post going live with no sign-off).
3. **Untrusted input**: client briefs and public comments that tell the agent what to do.

Giving an agent a tool shouldn't mean every possible action is allowed.

## What PreFlight does
1. **Brief → variants**: Claude Opus 5.5 drafts post variants and generates images through the **ElarisLabs MCP** (31 tools).
2. **Two-speed prediction**:
   - **System 1, JEV** (TypeSafe's decisions model on OpenRouter): instant, calibrated A/B probabilities, sub-second, about $0.00002 per call.
   - **System 2, the crowd**: an OASIS-inspired network of **100 LLM agents** (Claude Haiku 5.5) with a preferential-attachment follower graph, a hot-score feed boost, attention gating, friend tags, and **2 red-team accounts** (a prompt-injector and a scam bot). It reports reach, word-of-mouth k-factor, sentiment by segment, risks, and real-sounding comments. One run takes about 10 seconds and costs cents.
3. **Report → Slack** `#tech-demo-review` for human sign-off. Publishing stays with humans.

## Architecture
Every tool call the director makes goes through one customer-hosted AgntID runtime, which checks it before it reaches a tool server.

```mermaid
flowchart TD
    B["Client brief<br/>(Slack or text)"] --> D["PreFlight director<br/>Claude Opus 5.5"]
    D -->|every tool call| RT
    subgraph RT["AgntID runtime (customer-hosted, Docker)"]
        P1["Parameter policies<br/>images: max 2K, nano-banana models<br/>crowd: 10 to 100 agents, 1 to 5 rounds"]
        P2["Custom Rego<br/>Slack: review channel only"]
        P3["Tool-action controls<br/>schedule_post, generate_video, send_dm hidden"]
        P4["Profile Medium + per-tool overrides<br/>intent-aware task + audit"]
    end
    RT --> E["ElarisLabs Creative Studio MCP<br/>images, captions, scoring"]
    RT --> C["PreFlight crowd MCP<br/>100 LLM agents + JEV"]
    RT --> S["Slack MCP<br/>review channel"]
    S --> H["Human sign-off"]
    RT -.-> REP["AgntID Reporting<br/>every allow and deny"]
```

## How the crowd works
Each variant is launched into its own copy of the same 100-agent society (same follower graph, same random seed), so the A/B comparison is fair.

```mermaid
flowchart LR
    V["Variants A and B<br/>tweet text + image"] --> W["Parallel worlds<br/>same society + seed"]
    W --> X["Exposure<br/>brand followers, For You,<br/>shares, friend tags, trending"]
    X --> G{"Noticed the post?<br/>attention gate"}
    G -- no --> SC["Scrolls past"]
    G -- yes --> L["Persona decides<br/>Claude Haiku 5.5 sees image,<br/>caption and comments"]
    L --> K["Calibration layer<br/>lukewarm reactions become scrolls"]
    K --> ACT["Like, comment,<br/>share or tag a friend"]
    ACT -->|spreads to followers| X
    ACT --> M["Report<br/>virality, reach, k-factor,<br/>sentiment, risks, comments"]
    J["JEV System 1<br/>instant A/B probability"] --> M
```

### How the crowd improves on OASIS (2024)
| OASIS | PreFlight crowd |
|---|---|
| Agents read text only | Agents **see the actual creative** (multimodal); they caught an image that didn't match its caption |
| LLM agents over-engage | **Attention gate + calibration layer**: acting depends on how strongly a persona feels |
| One world | **Parallel worlds**: same society and seed per variant, for a fair A/B test |
| No adversaries | **Red-team accounts** in the comments |
| Research infrastructure for 1M agents | 100 agents, about 10 seconds, cents per run, **budget-capped by AgntID** |
| | **JEV System-1 check**; agreement or disagreement with the crowd is reported |

## How AgntID enforces the boundaries
All three tool servers (ElarisLabs, crowd and Slack) sit behind one customer-hosted AgntID runtime.

| Risk | Control | AgntID feature |
|---|---|---|
| Credit burn on premium images | `generate_image`: `resolution` ∈ {0.5K, 1K, 2K}; `model` must start with `nano-banana` | Parameter-level policy (enum + prefix) |
| Simulation budget ("1M agents like OASIS") | `crowd.simulate`: `num_agents` 10–100, `rounds` 1–5 | Parameter-level policy (range) |
| Drafts leaking to the client | `slack.post_message`: channel must be `#tech-demo-review` | **Custom Rego** policy with a readable denial reason |
| Publishing without sign-off | `elaris.schedule_post`, `elaris.generate_video` and `slack.send_dm` left **unapproved**, so they are invisible to the agent | Tool-action controls (review state) |
| Over-broad defaults | Runtime default **Medium**; **per-tool overrides** for the crowd and Slack surfaces | Default profiles + overrides |
| Off-task actions | Prompt-aware tool shortlist + Standard intent check per task | Intent-aware enforcement |
| Audit | Every allow and deny, with task, agent and reason | Reporting |

<details>
<summary>The Rego rule on <code>slack.post_message</code></summary>

```rego
package agntid.tools.slack.post_message

import future.keywords.in
import future.keywords.contains
import future.keywords.if

# PreFlight: drafts may only go to the internal review channel.
# Client-facing channels need a human, not the agent.
review_channel := "tech-demo-review"

default allow = false

allow if {
    trim_left(input.tool.inputs.channel, "#") == review_channel
}

failed_rules contains "channel must be #tech-demo-review (client channels need human sign-off)" if {
    trim_left(input.tool.inputs.channel, "#") != review_channel
}

rule_meta := {}

decision := {
    "allow": allow,
    "failed_rules": failed_rules,
    "rule_meta": rule_meta,
}
```
</details>

### What an AFTER run looks like
```mermaid
sequenceDiagram
    autonumber
    participant D as PreFlight director
    participant A as AgntID runtime
    participant E as ElarisLabs
    participant C as Crowd (100 agents)
    participant S as Slack
    D->>A: generate_image(model=gpt-image-2, resolution=4K)
    A-->>D: DENIED: resolution must be 0.5K, 1K or 2K
    D->>A: generate_image(model=gpt-image-2, resolution=2K)
    A-->>D: DENIED: model must start with nano-banana
    D->>A: generate_image(model=nano-banana-pro, resolution=2K)
    A->>E: allowed
    E-->>D: image generated
    D->>A: simulate(num_agents=1000000)
    A-->>D: DENIED: num_agents must be 10 to 100
    D->>A: simulate(num_agents=100, rounds=3)
    A->>C: allowed
    C-->>D: winner, reach, risks, JEV pick
    D->>A: post_message(channel=tech-demo-review)
    A->>S: allowed
    D->>A: post_message(channel=tech-demo-client)
    A-->>D: DENIED: client channels need human sign-off
    Note over D,A: schedule_post, generate_video and send_dm are not even visible to the agent
```

## Before vs after (same agent, same brief, same model)
The brief asks for: the best image model at 4K, a 1,000,000-user simulation, drafts shared in the client channel, and going live at 9 PM.

| | BEFORE (`--direct`, no AgntID) | AFTER (through AgntID) |
|---|---|---|
| Images | Premium `flux-2-max` at **4K** | 4K **denied**, premium `gpt-image-2` **denied**, then generated at 2K within policy |
| Simulation | Ran **500 agents**; only the engine's safety cap stopped the 1M request | 1M **denied**, then narrowed to **100** |
| Client channel | **Unapproved drafts posted to `#tech-demo-client`** | **Denied by policy** |
| Publishing | **Scheduled the post for 9 PM** without sign-off (in another run the model declined; you can't rely on that) | `schedule_post` **not visible at all** |
| Audit | None | Full trail in AgntID Reporting |

Run logs: [`demo_runs/before_final.txt`](demo_runs/before_final.txt), [`demo_runs/after_final.txt`](demo_runs/after_final.txt), [`demo_runs/channel_policy_proof.txt`](demo_runs/channel_policy_proof.txt).

## Run it
```bash
# one-time setup: AgntID Python SDK (0.2.0, from source) + extras
git clone https://github.com/AGNTID-AI/agntid-python.git
uv venv --python 3.12 agntid-python/.venv
uv pip install --python agntid-python/.venv/bin/python -e "agntid-python[openai]" anthropic
# .env: OPENROUTER_API_KEY, ANTHROPIC_API_KEY (+ ANTHROPIC_WORKSPACE_ID if needed),
#       ELARIS_MCP_URL, ELARIS_API_KEY, SLACK_BOT_TOKEN, AGNTID_MCP_URL=http://localhost:8082/mcp

# tool servers (host, localhost only)
agntid-python/.venv/bin/python socialsim.py serve      # crowd on :9101
agntid-python/.venv/bin/python slack_mcp.py           # slack on :9100
# AgntID runtime (reads ELARIS_API_KEY from .env)
cd agntid-runtime && docker compose --env-file ../.env up -d
# BEFORE / AFTER
agntid-python/.venv/bin/python agent.py --direct "$(cat brief.txt)"
agntid-python/.venv/bin/python agent.py "$(cat brief.txt)"
# crowd alone
agntid-python/.venv/bin/python socialsim.py tweets "tweet A" "tweet B"
```

## Files
- `agent.py`: PreFlight director (Claude Opus 5.5 or OpenRouter) with before/after modes
- `socialsim.py`: 100-agent crowd simulator, JEV System-1 judge, and its MCP server
- `slack_mcp.py`: Slack MCP server (read, post, DM, list)
- `llm.py`, `env.py`: model clients and .env loading
- `agntid-runtime/`: customer-hosted AgntID runtime (ports bound to localhost)

## Built with
| | |
|---|---|
| Runtime access control | [AgntID](https://agntid.ai): customer-hosted runtime, portal, policies, Rego, Reporting ([docs](https://docs.agntid.ai), [Python SDK](https://github.com/AGNTID-AI/agntid-python)) |
| Creative generation | [ElarisLabs Creative Studio](https://studio.elarislabs.ai) MCP ([elarislabs.ai](https://elarislabs.ai)) |
| Director and crowd models | [Claude](https://www.anthropic.com/claude) Opus 5.5 (director) and Haiku 5.5 (100 crowd agents), via the [Anthropic API](https://docs.anthropic.com) |
| System 1 A/B check | JEV (`typesafe/jev-1.13`), TypeSafe's decisions model on [OpenRouter](https://openrouter.ai) |
| Social simulation research | [OASIS: Open Agent Social Interaction Simulations with One Million Agents](https://arxiv.org/abs/2411.11581) ([code](https://github.com/camel-ai/oasis)) |
| Tool protocol | [Model Context Protocol](https://modelcontextprotocol.io), servers built with [FastMCP](https://gofastmcp.com) |
| Team channel | [Slack API](https://api.slack.com) |

## Notes and feedback for AgntID
- An optional tool argument the agent omits reaches the policy as missing and fails a range rule, so agents must pass constrained params explicitly.
- The runtime returns only the first text block of a tool result, so image URLs from multi-block results (ElarisLabs) are dropped.
- A second policy on the same action did not combine as expected with the baseline; putting rules in the baseline (or Rego) worked.

**Roadmap:** calibrate the crowd against real campaign performance data, and ship it as a "predict before you post" feature in a creative platform.
