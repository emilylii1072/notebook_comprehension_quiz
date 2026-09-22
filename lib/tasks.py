"""The study's task screens: what the participant is shown, and the countdown.

Three kinds of thing live here:

  * the task vocabulary -- which instruction document belongs to which screen
    (the main task has one per condition; ideation and debugging are shared),
    and how long each timed task is allowed;
  * `render_countdown`, the clock the participant watches;
  * `format_duration`, used by Admin to report what actually happened.

Instruction content is admin-authored (Admin > Task instructions, stored in the
`task_instructions` table) -- nothing here is study content, only the frame
around it.
"""

import base64
import html
import time

import streamlit as st
import streamlit.components.v1 as components

# Timed-task keys, as stored in participant_task_timings.task_key.
MAIN, IDEATE, DEBUG = "main", "ideate", "debug"

# Allowance per task, in seconds. None = untimed (measured, not limited).
TASK_LIMIT_SECONDS: dict[str, int | None] = {
    MAIN: 40 * 60,
    IDEATE: None,
    DEBUG: 15 * 60,
}

TASK_TITLE = {
    MAIN: "Main task",
    IDEATE: "Idea generation task",
    DEBUG: "Debugging task",
}

# task_instructions.task_key for the main task's one shared brief (the task
# itself -- data, goal, deliverable -- the same for every condition).
MAIN_BRIEF_KEY = "main_task_brief"

# task_instructions.task_key for the main task's condition instructions (how to
# go about it), per assigned condition. The participant sees the shared brief
# plus the one condition document matching their condition, one after another
# on the same page -- see instruction_keys_for / render_instructions.
MAIN_INSTRUCTION_KEY = {
    "slow_planning": "main_slow_planning",
    "slow_iterating": "main_slow_iterating",
    "control": "main_control",
}

# Every instruction document Admin can upload, in the order it's presented.
INSTRUCTION_KEYS: list[tuple[str, str]] = [
    (MAIN_BRIEF_KEY, "Main task — Task brief (shown to every condition)"),
    ("main_slow_planning", "Main task — Condition instructions: Slow planning"),
    ("main_slow_iterating", "Main task — Condition instructions: Slow iterating"),
    ("main_control", "Main task — Condition instructions: Control"),
    ("ideate", "Idea generation task"),
    ("debug", "Debugging task"),
]
INSTRUCTION_TITLE = dict(INSTRUCTION_KEYS)

# Admin-uploaded grading/reference material, kept in task_instructions beside
# the participant-facing documents but deliberately not in INSTRUCTION_KEYS (a
# task can start without it). DEBUG_NOTEBOOK_KEY: the buggy notebook (title =
# the uploaded filename) -- its flattened text (lib.notebook.notebook_to_text)
# is what the grader reads each write-up against; its rendered HTML
# (lib.notebook.notebook_to_html, `notebook_html` column) is what
# render_notebook_link offers the participant to view during the debugging task.
DEBUG_NOTEBOOK_KEY = "debug_notebook"
# The rubric the ideation pitch transcript is scored against, verbatim.
IDEATE_RUBRIC_KEY = "ideate_rubric"


def instruction_keys_for(task_key: str, condition: str | None) -> list[str]:
    """Which uploaded document(s) a given task screen should show, in the order
    they're shown. The main task shows two: the shared brief, then the
    condition instructions matching the participant's assigned condition.
    Ideation and debugging each show their one shared document."""
    if task_key == MAIN:
        return [MAIN_BRIEF_KEY, MAIN_INSTRUCTION_KEY.get(condition or "")]
    return [task_key] if task_key in (IDEATE, DEBUG) else []


def format_duration(seconds: float | None) -> str:
    """'12m 03s' / '1h 04m' / '—'. Used in Admin, not in the countdown."""
    if seconds is None:
        return "—"
    seconds = int(round(float(seconds)))
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{sign}{h}h {m:02d}m"
    return f"{sign}{m}m {s:02d}s"


def render_countdown(*, started_at_epoch: float, limit_seconds: int | None) -> None:
    """The participant's clock, counting from a server-supplied start.

    Rendered as one self-contained iframe so it ticks on its own -- Streamlit
    never reruns to advance it, and a page refresh doesn't restart it, because
    the start time comes from the database rather than from the browser.

    An untimed task counts up. A timed one counts down, then beeps twice and
    keeps counting into overtime: time being up is a prompt to wrap up, not a
    lock on the task, so nothing here disables anything.

    The beep needs an AudioContext, which the browser will only start off a user
    gesture. Clicking "Start" in Streamlit is that gesture and normally carries
    into this same-origin frame; when it doesn't, the frame notices the context
    is suspended and offers a button to enable sound rather than failing silently.
    """
    elapsed = max(0.0, time.time() - started_at_epoch)
    counts_down = limit_seconds is not None
    remaining = (limit_seconds - elapsed) if counts_down else elapsed

    doc = _COUNTDOWN_HTML.replace("__REMAINING__", f"{remaining:.1f}").replace(
        "__COUNTS_DOWN__", "true" if counts_down else "false"
    ).replace(
        "__LIMIT_LABEL__",
        html.escape(format_duration(limit_seconds)) if counts_down else "",
    )
    # components.html, not st.iframe, despite the deprecation notice: st.iframe
    # takes a src rather than markup, and pointing it at a data: URL would put
    # the clock in an opaque origin, where the user activation that lets the
    # beep play does not reach it. A srcdoc frame stays same-origin.
    components.html(doc, height=232)


_COUNTDOWN_HTML = """
<!doctype html>
<meta charset="utf-8">
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; font-family: "Source Sans Pro", system-ui, -apple-system, sans-serif; }
  .wrap { text-align: center; padding: 10px 10px 4px; }
  .clock { font-size: 56px; font-weight: 700; letter-spacing: 1px; line-height: 1.05;
           font-variant-numeric: tabular-nums; color: #1baf7a; }
  .clock.warn { color: #e0913f; }
  .clock.over { color: #c1462f; }
  .label { margin-top: 6px; font-size: 14px; opacity: .75; }
  .msg { margin-top: 8px; font-size: 15px; line-height: 1.3; font-weight: 600; color: #c1462f; min-height: 20px; }
  button { margin-top: 8px; font-size: 13px; padding: 6px 12px; border-radius: 8px;
           border: 1px solid #8a8a8a; background: transparent; color: inherit; cursor: pointer; }
</style>
<div class="wrap">
  <div id="clock" class="clock">--:--</div>
  <div class="label" id="label"></div>
  <div class="msg" id="msg"></div>
  <button id="sound" style="display:none">🔔 Enable the alarm sound</button>
</div>
<script>
(function () {
  var remaining   = parseFloat("__REMAINING__");
  var countsDown  = __COUNTS_DOWN__;
  var limitLabel  = "__LIMIT_LABEL__";
  var startedAt   = Date.now();
  var clockEl = document.getElementById("clock");
  var labelEl = document.getElementById("label");
  var msgEl   = document.getElementById("msg");
  var soundEl = document.getElementById("sound");
  var beeped  = false;
  var ctx = null;

  labelEl.textContent = countsDown ? ("of " + limitLabel + " remaining") : "elapsed";

  function audio() {
    if (ctx) return ctx;
    try { ctx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { ctx = null; }
    return ctx;
  }
  // One 880Hz blip, shaped so it doesn't click.
  function blip(at) {
    var c = audio(); if (!c) return;
    var osc = c.createOscillator(), gain = c.createGain();
    osc.type = "sine"; osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.0001, at);
    gain.gain.exponentialRampToValueAtTime(0.35, at + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.32);
    osc.connect(gain); gain.connect(c.destination);
    osc.start(at); osc.stop(at + 0.34);
  }
  // resume() is asynchronous -- reading ctx.state straight after it still says
  // "suspended" -- so the outcome comes back through a callback rather than a
  // return value. Getting this wrong means the fallback button never works.
  function beepBeep(done) {
    done = done || function () {};
    var c = audio();
    if (!c) { done(false); return; }
    function ring() {
      if (c.state !== "running") { done(false); return; }
      for (var round = 0; round < 3; round++) {
        blip(c.currentTime + round * 1.0);
        blip(c.currentTime + round * 1.0 + 0.42);
      }
      done(true);
    }
    if (c.state === "suspended") {
      var p = c.resume();
      if (p && p.then) { p.then(ring, function () { done(false); }); } else { ring(); }
    } else {
      ring();
    }
  }
  soundEl.addEventListener("click", function () {
    beepBeep(function (ok) { if (ok) { soundEl.style.display = "none"; } });
  });

  function fmt(total) {
    var neg = total < 0;
    total = Math.abs(Math.floor(total));
    var h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
    var body = (h > 0 ? String(h) + ":" + String(m).padStart(2, "0")
                      : String(m)) + ":" + String(s).padStart(2, "0");
    return (neg ? "+" : "") + body;
  }

  function tick() {
    var gone = (Date.now() - startedAt) / 1000;
    var value = countsDown ? (remaining - gone) : (remaining + gone);
    clockEl.textContent = fmt(value);
    if (countsDown) {
      if (value <= 0) {
        clockEl.className = "clock over";
        labelEl.textContent = "over the " + limitLabel + " allowance";
        msgEl.textContent = "Time is up — please wrap up and upload your files below.";
        if (!beeped) {
          beeped = true;
          beepBeep(function (ok) {
            if (!ok) { soundEl.style.display = "inline-block"; }
          });
        }
      } else if (value <= 120) {
        clockEl.className = "clock warn";
      }
    }
  }
  tick();
  setInterval(tick, 1000);
})();
</script>
"""


def render_instructions(instructions: list[dict | None], task_key: str) -> bool:
    """Show one or more admin-uploaded task documents, in order, one after
    another on the same page (the main task shows its shared brief followed by
    the participant's condition instructions; ideation and debugging each show
    their one shared document). Returns False (and says so, rendering nothing)
    when any of them hasn't been uploaded yet, so the caller can refuse to
    start a task the participant can't actually read."""
    if any(doc is None or not (doc.get("content") or "").strip() for doc in instructions):
        st.error(
            f"The {TASK_TITLE.get(task_key, task_key)} instructions haven't been "
            "uploaded yet. Please tell the researcher before continuing."
        )
        return False
    for doc in instructions:
        title = (doc.get("title") or "").strip()
        if title:
            st.markdown(f"### {title}")
        st.markdown(doc["content"])
    return True


def render_notebook_link(nb_row: dict | None) -> None:
    """The Debugging task's "open the buggy notebook in a new tab" control -- the
    .ipynb participants are given, rendered to a self-contained HTML page
    (lib.notebook.notebook_to_html) and linked via a data: URI so no server route
    is needed. If the admin hasn't uploaded a notebook yet, or uploaded one before
    this feature existed (no HTML saved for it), this says so instead of silently
    rendering nothing -- a participant who can't see this needs to know it's
    supposed to be there, not just miss it."""
    nb_html = (nb_row or {}).get("notebook_html")
    if not nb_html:
        st.warning(
            "The buggy notebook isn't available to view yet. Please tell the "
            "researcher before continuing."
        )
        return
    b64 = base64.b64encode(nb_html.encode("utf-8")).decode("ascii")
    st.markdown(
        f'📓 <a href="data:text/html;base64,{b64}" target="_blank" rel="noopener">'
        "<strong>Open the buggy notebook in a new tab</strong></a>",
        unsafe_allow_html=True,
    )
    st.caption("Keep it open alongside this page while you look for bugs.")
