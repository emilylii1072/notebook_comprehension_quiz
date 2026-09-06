"""Parse a Claude Code session JSONL transcript, chart it as a swimlane timeline,
and derive per-session behavioural metrics.

Best-effort parser for the Claude Code transcript format: each line is a JSON object
with a `type` ("user" | "assistant" | other), a `message` (role + content, following
the Anthropic Messages API shape), and a `timestamp`. A "turn" starts at each real user
prompt (a user message that isn't purely a tool_result wrapper) and runs until the next
one. Unrecognized/malformed lines are skipped and counted, not fatal.

`parse_jsonl` / `build_figure` / `show_event_dialog` are lifted unchanged from the old
Session Timeline page; `compute_log_metrics` is new and feeds the admin's per-participant
view and the cross-condition statistics.
"""

import json
import statistics
import string
from datetime import datetime, timedelta, timezone

import plotly.graph_objects as go
import streamlit as st

LANE_ORDER = ["Tool Call", "Agent Call", "User Prompt"]  # bottom -> top
LANE_Y = {"Tool Call": 0, "Agent Call": 1, "User Prompt": 2}

COLOR_USER = "#2a78d6"    # categorical slot 1 (blue)
COLOR_AGENT = "#1baf7a"   # categorical slot 2 (aqua)
COLOR_TOOL = "#eda100"    # categorical slot 3 (yellow)
COLOR_MUTED = "#898781"   # muted ink (connector guide lines)
COLOR_RING = "#fcfcfb"    # light chart surface, used as the marker ring

# Tools that mutate a file — the count of these is the participant's "edit"/iteration
# volume in the cohort stats.
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}

# Distinct Plotly marker symbols, assigned in order to whatever tool names actually
# appear in the uploaded session (so the legend adapts to the real data instead of
# guessing every possible tool name up front).
TOOL_SYMBOL_SEQUENCE = [
    "diamond", "square", "triangle-up", "star", "hexagon", "pentagon",
    "cross", "x", "triangle-down", "hourglass", "bowtie", "diamond-cross",
]
PROMPT_SYMBOL = {"Instruct": "circle", "Chat": "star"}

# --- Chat vs. Instruct heuristic -------------------------------------------------
# A prompt is classified purely from surface phrasing -- no API call. This keeps the
# tool fully offline. Rules, in order:
#   1. Ends with "?", or opens with a question/auxiliary word  -> Chat
#   2. Opens with a common imperative/command verb              -> Instruct
#   3. Otherwise, default to Instruct -- in an agentic coding transcript most
#      declarative turns ("the tests are failing", "this looks wrong") are implicit
#      requests for action even without a leading command verb.
#   4. Override (applied after parsing, once tool usage is known): if the turn this
#      prompt started actually invoked a tool, it's Instruct regardless of the text
#      heuristic -- e.g. "Can you check if the tests pass?" reads like a question but
#      produced tool calls, so it's actionable. See the post-pass in parse_jsonl().
QUESTION_WORDS = {
    "what", "how", "why", "when", "where", "who", "which", "whose",
    "can", "could", "would", "should", "is", "are", "was", "were",
    "do", "does", "did", "will", "have", "has", "had",
}
INSTRUCT_VERBS = {
    "fix", "add", "write", "create", "update", "remove", "delete", "refactor",
    "implement", "build", "change", "run", "generate", "make", "rename", "move",
    "install", "configure", "set", "setup", "debug", "optimize", "test", "check",
    "review", "clean", "improve", "revert", "undo", "commit", "push", "merge",
    "deploy", "upgrade", "downgrade", "rewrite", "extract", "split", "combine",
    "please", "let's", "lets", "start", "stop", "restart", "list", "show",
    "explain", "summarize", "document", "comment", "use", "switch", "convert",
}


def classify_prompt(text: str) -> str:
    t = text.strip()
    if not t:
        return "Chat"
    if t.endswith("?"):
        return "Chat"
    first_word = t.split()[0].lower().strip(string.punctuation) if t.split() else ""
    if first_word in QUESTION_WORDS:
        return "Chat"
    if first_word in INSTRUCT_VERBS:
        return "Instruct"
    return "Instruct"


# --- Parsing ----------------------------------------------------------------------

def _parse_timestamp(raw) -> float | None:
    """Parse an ISO 8601 timestamp string into a Unix epoch float, or None."""
    if not raw:
        return None
    try:
        text = raw.replace("Z", "+00:00") if isinstance(raw, str) else raw
        return datetime.fromisoformat(text).timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


def _message_content(record: dict):
    return (record.get("message") or {}).get("content")


def _is_tool_result_only(content) -> bool:
    """True if a user-message's content consists entirely of tool_result blocks
    (i.e. it's a tool response, not a genuine new prompt from the person)."""
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def _is_real_user_prompt(record: dict) -> bool:
    if record.get("type") != "user":
        return False
    content = _message_content(record)
    if isinstance(content, str):
        return content.strip() != ""
    if isinstance(content, list):
        return bool(content) and not _is_tool_result_only(content)
    return False


def _text_of(content) -> str:
    """Join every text block in a message's content into one string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return ""


def _preview(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _assign_declump_offsets(events: list[dict]) -> None:
    """Spread out events that share a turn+lane with a small vertical "y_offset",
    so a burst of near-simultaneous events (e.g. several tool calls from one
    assistant turn) renders as a small readable cluster instead of one overlapping
    blob. The offset always fits inside the lane's own band (+/- 0.35) no matter
    how many events are in the group.
    """
    groups: dict[tuple[int, str], list[dict]] = {}
    for e in events:
        groups.setdefault((e["turn"], e["lane"]), []).append(e)
    for group in groups.values():
        group.sort(key=lambda e: (e["t"], e["idx"]))
        n = len(group)
        step = min(0.08, 0.7 / (n - 1)) if n > 1 else 0.0
        for i, e in enumerate(group):
            e["y_offset"] = (i - (n - 1) / 2) * step


def parse_jsonl(raw_text: str) -> dict:
    """Parse a Claude Code session JSONL transcript into a flat list of lane events.

    Returns a dict with:
      - "events": list of event dicts, each carrying at least
        {"idx", "turn", "lane", "t", "label", "record"}, plus lane-specific fields.
      - "skipped": count of lines that were malformed JSON or an unrecognized shape.
      - "t0": epoch seconds of the first usable event (for relative-time display).

    Event fields by lane:
      User Prompt : "text", "classification" ("Chat" | "Instruct")
      Agent Call  : "text", "usage" (dict of token counts, may be empty)
      Tool Call   : "tool_name", "input" (dict), "result_text", "usage" (tokens of
                    the parent Agent Call that invoked this tool)
    """
    events: list[dict] = []
    skipped = 0
    turn_no = 0
    pending_tool_events: dict[str, int] = {}

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(record, dict):
            skipped += 1
            continue

        rtype = record.get("type")
        ts = _parse_timestamp(record.get("timestamp"))

        if _is_real_user_prompt(record):
            turn_no += 1
            text = _text_of(_message_content(record))
            if ts is not None:
                events.append({
                    "idx": len(events), "turn": turn_no, "lane": "User Prompt",
                    "t": ts, "label": _preview(text), "record": record,
                    "text": text, "classification": classify_prompt(text),
                })
            continue

        if turn_no == 0:
            skipped += 1
            continue

        if rtype == "assistant":
            content = _message_content(record) or []
            text = _text_of(content)
            usage = (record.get("message") or {}).get("usage") or {}
            if ts is not None:
                events.append({
                    "idx": len(events), "turn": turn_no, "lane": "Agent Call",
                    "t": ts, "label": _preview(text) or "(tool call only)",
                    "record": record, "text": text, "usage": usage,
                })
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name", "tool")
                        if ts is not None:
                            events.append({
                                "idx": len(events), "turn": turn_no, "lane": "Tool Call",
                                "t": ts, "label": name, "record": record,
                                "tool_name": name, "input": block.get("input", {}),
                                "result_text": "", "usage": usage,
                            })
                            pending_tool_events[block.get("id", "")] = len(events) - 1
            continue

        if rtype == "user" and _is_tool_result_only(_message_content(record)):
            content = _message_content(record)
            for block in content:
                tool_use_id = block.get("tool_use_id")
                target_idx = pending_tool_events.get(tool_use_id)
                if target_idx is not None:
                    c = block.get("content")
                    events[target_idx]["result_text"] = (
                        c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                    )
            continue

        skipped += 1

    turns_with_tools = {e["turn"] for e in events if e["lane"] == "Tool Call"}
    for e in events:
        if e["lane"] == "User Prompt" and e["turn"] in turns_with_tools:
            e["classification"] = "Instruct"

    _assign_declump_offsets(events)

    t0 = min((e["t"] for e in events), default=None)
    return {"events": events, "skipped": skipped, "t0": t0}


# --- Derived metrics ------------------------------------------------------------

def compute_log_metrics(parsed: dict) -> dict:
    """Reduce a parsed session into the scalar behavioural metrics the admin views
    use. Safe on an empty/degenerate session: every field is present, zeros/None
    where undefined.

    Manipulation-check metrics:
      time_to_first_tool_call_s — planning time before the agent first acted;
        expected highest under the "slow planning" condition.
      median_inter_tool_gap_s   — typical pause between successive tool calls;
        expected highest under the "slow iterating" condition.
    """
    events = parsed.get("events", [])
    t0 = parsed.get("t0")

    prompts = [e for e in events if e["lane"] == "User Prompt"]
    agents = [e for e in events if e["lane"] == "Agent Call"]
    tools = sorted((e for e in events if e["lane"] == "Tool Call"), key=lambda e: e["t"])

    tool_counts: dict[str, int] = {}
    for e in tools:
        tool_counts[e["tool_name"]] = tool_counts.get(e["tool_name"], 0) + 1

    tool_times = [e["t"] for e in tools]
    inter_gaps = [b - a for a, b in zip(tool_times, tool_times[1:])] if len(tool_times) > 1 else []

    all_times = [e["t"] for e in events]
    duration = (max(all_times) - min(all_times)) if all_times else 0.0

    time_to_first_tool = None
    if tool_times and t0 is not None:
        first_prompt_t = min((p["t"] for p in prompts), default=t0)
        time_to_first_tool = round(tool_times[0] - first_prompt_t, 1)

    tokens_in = sum((a["usage"] or {}).get("input_tokens", 0) or 0 for a in agents)
    tokens_out = sum((a["usage"] or {}).get("output_tokens", 0) or 0 for a in agents)
    cache_read = sum((a["usage"] or {}).get("cache_read_input_tokens", 0) or 0 for a in agents)

    return {
        "session_duration_s": round(duration, 1),
        "n_turns": len({e["turn"] for e in events}),
        "n_user_prompts": len(prompts),
        "n_agent_calls": len(agents),
        "n_tool_calls": len(tools),
        "n_edits": sum(v for k, v in tool_counts.items() if k in EDIT_TOOLS),
        "tool_counts": tool_counts,
        "time_to_first_tool_call_s": time_to_first_tool,
        "median_inter_tool_gap_s": round(statistics.median(inter_gaps), 1) if inter_gaps else None,
        "mean_inter_tool_gap_s": round(statistics.fmean(inter_gaps), 1) if inter_gaps else None,
        "chat_count": sum(1 for p in prompts if p["classification"] == "Chat"),
        "instruct_count": sum(1 for p in prompts if p["classification"] == "Instruct"),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cache_read_tokens": cache_read,
        "skipped_lines": parsed.get("skipped", 0),
    }


# --- Charting -----------------------------------------------------------------

def build_figure(parsed: dict) -> go.Figure:
    events = parsed["events"]
    t0 = parsed["t0"] or 0.0

    def rel(t: float) -> datetime:
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=t - t0)

    fig = go.Figure()

    turns = sorted({e["turn"] for e in events})
    guide_x, guide_y, turn_x, turn_text = [], [], [], []
    for turn in turns:
        prompt_events = [e for e in events if e["turn"] == turn and e["lane"] == "User Prompt"]
        if not prompt_events:
            continue
        start = rel(prompt_events[0]["t"])
        guide_x += [start, start, None]
        guide_y += [LANE_Y["Tool Call"], LANE_Y["User Prompt"], None]
        turn_x.append(start)
        turn_text.append(f"T{turn}")

    fig.add_trace(go.Scatter(
        x=guide_x, y=guide_y, mode="lines",
        line=dict(color=COLOR_MUTED, width=1, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=turn_x, y=[LANE_Y["User Prompt"] + 0.35] * len(turn_x),
        mode="text", text=turn_text, textposition="middle center",
        textfont=dict(color=COLOR_MUTED, size=11),
        hoverinfo="skip", showlegend=False,
    ))

    def y_of(e: dict) -> float:
        return LANE_Y[e["lane"]] + e.get("y_offset", 0.0)

    agent_events = [e for e in events if e["lane"] == "Agent Call"]
    fig.add_trace(go.Scatter(
        x=[rel(e["t"]) for e in agent_events], y=[y_of(e) for e in agent_events],
        mode="markers", name="Agent Call",
        marker=dict(size=12, color=COLOR_AGENT, symbol="square",
                    line=dict(width=2, color=COLOR_RING)),
        customdata=[e["idx"] for e in agent_events],
        hovertext=[f"T{e['turn']} · Agent Call<br>{e['label']}" for e in agent_events],
        hoverinfo="text",
    ))

    for kind in ("Instruct", "Chat"):
        sub = [e for e in events if e["lane"] == "User Prompt" and e["classification"] == kind]
        fig.add_trace(go.Scatter(
            x=[rel(e["t"]) for e in sub], y=[y_of(e) for e in sub],
            mode="markers", name=f"User Prompt ({kind})",
            marker=dict(size=12, color=COLOR_USER, symbol=PROMPT_SYMBOL[kind],
                        line=dict(width=2, color=COLOR_RING)),
            customdata=[e["idx"] for e in sub],
            hovertext=[f"T{e['turn']} · {kind}<br>{e['label']}" for e in sub],
            hoverinfo="text",
        ))

    tool_events = [e for e in events if e["lane"] == "Tool Call"]
    tool_names = sorted({e["tool_name"] for e in tool_events})
    symbol_for_tool = {
        name: TOOL_SYMBOL_SEQUENCE[i % len(TOOL_SYMBOL_SEQUENCE)]
        for i, name in enumerate(tool_names)
    }
    for name in tool_names:
        sub = [e for e in tool_events if e["tool_name"] == name]
        fig.add_trace(go.Scatter(
            x=[rel(e["t"]) for e in sub], y=[y_of(e) for e in sub],
            mode="markers", name=name,
            marker=dict(size=12, color=COLOR_TOOL, symbol=symbol_for_tool[name],
                        line=dict(width=2, color=COLOR_RING)),
            customdata=[e["idx"] for e in sub],
            hovertext=[f"T{e['turn']} · Tool Call<br>{name}" for e in sub],
            hoverinfo="text",
        ))

    fig.update_layout(
        yaxis=dict(
            tickmode="array", tickvals=[0, 1, 2], ticktext=LANE_ORDER,
            range=[-0.6, 2.9], showgrid=False, zeroline=False,
        ),
        xaxis=dict(title="Relative session time", tickformat="%H:%M:%S", showgrid=True),
        height=380,
        margin=dict(l=10, r=10, t=30, b=40),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    return fig


# --- Detail popup ---------------------------------------------------------------

def _format_tokens(usage: dict) -> str:
    if not usage:
        return "n/a"
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    if inp is None and out is None:
        return "n/a"
    parts = []
    if inp is not None:
        parts.append(f"{inp:,} in")
    if out is not None:
        parts.append(f"{out:,} out")
    cache_read = usage.get("cache_read_input_tokens")
    if cache_read:
        parts.append(f"{cache_read:,} cache read")
    return " · ".join(parts)


def _format_session_time(t: float, t0: float) -> str:
    rel_seconds = int(t - t0)
    mm, ss = divmod(rel_seconds, 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


@st.dialog("Event detail", width="large")
def show_event_dialog(event: dict, t0: float):
    session_time = _format_session_time(event["t"], t0)
    ts = datetime.fromtimestamp(event["t"], tz=timezone.utc)
    st.caption(
        f"Turn {event['turn']} · {event['lane']} · "
        f"Session time {session_time} · {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    if event["lane"] == "User Prompt":
        st.markdown(f"**Classification:** {event['classification']}")
        st.markdown("**Content**")
        st.markdown(event["text"] or "_(empty)_")
        st.markdown("**Tokens:** n/a (not reported on user turns)")
    elif event["lane"] == "Agent Call":
        st.markdown("**Content**")
        st.markdown(event["text"] or "_(no text -- tool call only)_")
        st.markdown(f"**Tokens:** {_format_tokens(event['usage'])}")
    else:  # Tool Call
        st.markdown(f"**Tool:** {event['tool_name']}")
        st.markdown("**Input**")
        st.json(event["input"])
        if event["result_text"]:
            st.markdown("**Result**")
            st.markdown(f"```\n{event['result_text'][:2000]}\n```")
        st.markdown(
            f"**Tokens (from the agent call that invoked this tool):** "
            f"{_format_tokens(event['usage'])}"
        )

    with st.expander("Raw JSON"):
        st.json(event["record"])


def render_timeline(raw_jsonl: str, key: str) -> None:
    """Render the swimlane chart for one session with click-to-open detail popups.
    `key` must be unique per participant so Streamlit keeps the charts distinct.
    """
    parsed = parse_jsonl(raw_jsonl)
    if not parsed["events"]:
        st.info("No recognizable turns in this session log.")
        return
    if parsed["skipped"]:
        st.caption(f"Skipped {parsed['skipped']} unparseable/unrecognized line(s).")

    event_state = st.plotly_chart(
        build_figure(parsed), on_select="rerun", selection_mode="points",
        key=f"timeline_{key}", width="stretch",
    )
    points = (event_state or {}).get("selection", {}).get("points", [])
    if points:
        clicked_idx = points[0].get("customdata")
        shown_key = f"_last_shown_event_{key}"
        if clicked_idx is not None and clicked_idx != st.session_state.get(shown_key):
            st.session_state[shown_key] = clicked_idx
            clicked_event = next(e for e in parsed["events"] if e["idx"] == clicked_idx)
            show_event_dialog(clicked_event, parsed["t0"])
