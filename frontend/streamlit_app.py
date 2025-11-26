"""
Seedling Labs – AI-Powered GitHub Issue Assistant (Frontend, Light UI)

This Streamlit app:

1. Takes a public GitHub repository URL + issue number from the user.
2. Sends them to the FastAPI backend (/analyze).
3. Displays:
   - Issue metadata (repo, title, GitHub link, stars, open issues).
   - AI analysis (summary, type, priority, suggested labels, impact).
   - Extra triage suggestion (who should handle it, how urgent).
4. Shows a history of recent analyses.
5. Optionally shows the raw JSON from the LLM and lets the user download it.

We intentionally keep styling simple and rely on Streamlit’s default look so
that the UI is stable and professional without layout bugs.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Backend URL – change this if your FastAPI backend is running somewhere else.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# GitHub REST API base – used to fetch repo metadata (stars, etc.)
GITHUB_API_BASE = "https://api.github.com"


# -----------------------------------------------------------------------------
# SMALL HELPER: BACKEND HEALTH CHECK
# -----------------------------------------------------------------------------

def check_backend_health() -> bool:
    """
    Ping the /health endpoint once to check if the FastAPI backend is alive.

    This is just to surface a clean status to the user at the top of the app.
    """
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return r.ok
    except requests.RequestException:
        return False


# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Seedling Labs – GitHub Issue Assistant",
    page_icon="🌱",
    layout="centered",
)

# Top-level title
st.title("Seedling Labs – GitHub Issue Assistant")

# Backend health indicator
backend_ok = check_backend_health()
if backend_ok:
    st.success(f"Backend connected at {BACKEND_URL}")
else:
    st.error(
        f"Backend not reachable at {BACKEND_URL}. "
        "Make sure FastAPI is running (e.g., `uvicorn main:app --reload`)."
    )

st.write("")  # small spacing


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------

# Current input values that the form widgets bind to
if "repo_url_input" not in st.session_state:
    st.session_state.repo_url_input = "https://github.com/facebook/react"

if "issue_input" not in st.session_state:
    st.session_state.issue_input = 25056  # known valid issue

# Recent history (for History tab)
if "history" not in st.session_state:
    st.session_state.history = []  # list of {repo_url, issue_number, issue_title, ai_analysis}

# Prefill values used when user clicks "load from history"
# We use separate keys so we can safely update them from the History tab,
# then copy them into the widget keys *before* the widgets are created.
if "prefill_repo_url" not in st.session_state:
    st.session_state.prefill_repo_url = None

if "prefill_issue_number" not in st.session_state:
    st.session_state.prefill_issue_number = None

# If history requested a prefill in the previous run, apply it now
# BEFORE creating any widgets that use repo_url_input / issue_input.
if st.session_state.prefill_repo_url is not None:
    st.session_state.repo_url_input = st.session_state.prefill_repo_url
    st.session_state.prefill_repo_url = None

if st.session_state.prefill_issue_number is not None:
    st.session_state.issue_input = st.session_state.prefill_issue_number
    st.session_state.prefill_issue_number = None


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def render_priority(priority_text: str) -> str:
    """
    Convert the AI's "priority_score" text into a friendly label with emoji.

    Example input:
        "4 - High: crashes devtools at start for many users."
    """
    score = None
    for ch in priority_text:
        if ch.isdigit():
            score = int(ch)
            break

    if score is None:
        return f"`{priority_text}`"

    if score <= 1:
        label = "Low"
        emoji = "🟢"
    elif score == 2:
        label = "Low–Medium"
        emoji = "🟢"
    elif score == 3:
        label = "Medium"
        emoji = "🟡"
    elif score == 4:
        label = "High"
        emoji = "🟠"
    else:
        label = "Critical"
        emoji = "🔴"

    return f"{emoji} **Priority {score} – {label}**  \n{priority_text}"


def parse_owner_repo(repo_url: str) -> Optional[Tuple[str, str]]:
    """
    Extract (owner, repo) from a GitHub URL.

    Example:
      https://github.com/facebook/react  -> ("facebook", "react")
    """
    if "github.com" not in repo_url:
        return None

    cleaned = repo_url.split("github.com/", 1)[-1]
    cleaned = cleaned.split("?", 1)[0].split("#", 1)[0].strip("/")
    parts = cleaned.split("/")

    if len(parts) < 2:
        return None

    owner, repo = parts[0], parts[1]
    if not owner or not repo:
        return None

    return owner, repo


def fetch_repo_metadata(repo_url: str) -> Optional[Dict[str, Any]]:
    """
    Call the public GitHub API to fetch repo info like stars and open issues.
    If it fails (rate limit, bad URL), we simply return None and skip it.
    """
    parsed = parse_owner_repo(repo_url)
    if not parsed:
        return None

    owner, repo = parsed
    api_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"

    try:
        resp = requests.get(api_url, timeout=10)
        if not resp.ok:
            return None
        return resp.json()
    except requests.RequestException:
        return None


def derive_triage_suggestion(issue_type: str, priority_text: str) -> str:
    """
    Small rule-based helper that suggests what to do with this issue
    based on type + priority score.
    """
    score = None
    for ch in priority_text:
        if ch.isdigit():
            score = int(ch)
            break
    if score is None:
        score = 3  # treat as medium

    if issue_type == "bug":
        if score >= 5:
            return "🚨 Critical bug – Page on-call engineer immediately and create a hotfix."
        elif score == 4:
            return "⚠️ High-priority bug – Assign to the current sprint and notify the owning team today."
        elif score == 3:
            return "🟡 Medium bug – Add to backlog and fix in the next sprint."
        else:
            return "🟢 Low bug – Keep in backlog; fix when there is spare capacity."
    elif issue_type == "feature_request":
        if score >= 4:
            return "💡 High-value feature – Share with Product Manager and consider for upcoming roadmap."
        else:
            return "✨ Nice-to-have – Keep in product backlog for future discussion."
    elif issue_type == "documentation":
        return "📘 Docs gap – Assign to docs/owning engineer; usually quick to fix."
    elif issue_type == "question":
        return "❓ Support / question – Reply in comments and convert to FAQ/docs if it repeats."
    else:
        return "🧩 Misc issue – Needs manual triage to decide owner and true priority."


# -----------------------------------------------------------------------------
# SIDEBAR: Branding + Sample Issues
# -----------------------------------------------------------------------------

with st.sidebar:
    st.subheader("GitHub Issue Assistant")
    st.write(
        "Use AI to quickly understand and prioritize GitHub issues.\n\n"
        "1. Paste a **public GitHub repo URL**\n"
        "2. Enter an **issue number**\n"
        "3. Get a structured summary, type, priority, labels, and impact."
    )

    st.divider()
    st.markdown("**Quick Demo Issues**")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("React #25056"):
            st.session_state.repo_url_input = "https://github.com/facebook/react"
            st.session_state.issue_input = 25056
            st.rerun()
    with col_b:
        if st.button("VS Code #1"):
            st.session_state.repo_url_input = "https://github.com/microsoft/vscode"
            st.session_state.issue_input = 1
            st.rerun()

    st.divider()
    st.caption(
        "Some repositories (like torvalds/linux) have issues disabled. "
        "In that case the backend returns a clear error."
    )


# -----------------------------------------------------------------------------
# MAIN TABS
# -----------------------------------------------------------------------------

tab_analyze, tab_history, tab_about = st.tabs(
    ["🔍 Analyze Issue", "🕒 History", "ℹ️ About"]
)


# -----------------------------------------------------------------------------
# TAB 1: ANALYZE ISSUE
# -----------------------------------------------------------------------------

with tab_analyze:
    st.header("Analyze a GitHub Issue")
    st.write(
        "Enter a public GitHub repository URL and an issue number. "
        "The backend will fetch the issue and comments, run them through an LLM, "
        "and return a structured analysis."
    )

    # Form so user can fill fields and submit in one click
    with st.form("issue_form", clear_on_submit=False):
        repo_url = st.text_input(
            "GitHub Repository URL",
            key="repo_url_input",
            help="Example: https://github.com/facebook/react",
        )

        issue_number = st.number_input(
            "Issue Number",
            min_value=1,
            step=1,
            format="%d",
            key="issue_input",
            help="Must be a positive integer.",
        )

        with st.expander("Advanced display options"):
            show_raw_json = st.checkbox(
                "Show raw JSON output section",
                value=True,
                help="Turn this off if you only care about the high-level analysis.",
            )

        submitted = st.form_submit_button("Analyze Issue 🚀")

    st.divider()

    if submitted:
        if not repo_url.strip():
            st.error("Please enter a valid GitHub repository URL.")
        else:
            payload: Dict[str, Any] = {
                "repo_url": repo_url.strip(),
                "issue_number": int(issue_number),
            }

            with st.spinner("Contacting backend and analyzing issue with the LLM..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/analyze",
                        json=payload,
                        timeout=90,
                    )

                    if not resp.ok:
                        try:
                            detail = resp.json().get("detail")
                        except Exception:
                            detail = resp.text
                        st.error(f"Backend error ({resp.status_code}): {detail}")
                    else:
                        data = resp.json()
                        analysis = data["ai_analysis"]

                        # Update recent history
                        st.session_state.history.insert(0, {
                            "repo_url": data["repo_url"],
                            "issue_number": data["issue_number"],
                            "issue_title": data["issue_title"],
                            "ai_analysis": analysis,
                        })
                        st.session_state.history = st.session_state.history[:5]

                        st.success("Analysis complete ✅")

                        # ----- Issue Metadata -----
                        st.subheader("Issue Metadata")
                        repo_meta = fetch_repo_metadata(data["repo_url"])

                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"**Repository:** `{data['repo_url']}`")
                            st.markdown(f"**Issue #:** `{data['issue_number']}`")
                            st.markdown(f"**Title:** {data['issue_title']}")
                            if data.get("issue_html_url"):
                                st.markdown(
                                    f"[View on GitHub]({data['issue_html_url']})",
                                    help="Open the original issue in a new tab.",
                                )
                        with col2:
                            if repo_meta:
                                owner = (repo_meta.get("owner") or {}).get("login", "")
                                stars = repo_meta.get("stargazers_count", 0)
                                open_issues = repo_meta.get("open_issues_count", 0)

                                st.markdown(f"**Owner:** `{owner}`")
                                st.markdown(f"⭐ **Stars:** `{stars}`")
                                st.markdown(f"📂 **Open issues:** `{open_issues}`")
                            else:
                                st.caption("Repo metadata unavailable (rate limit or invalid URL).")

                        st.divider()

                        # ----- AI Analysis -----
                        st.subheader("AI Analysis")

                        st.markdown(f"**Summary**  \n{analysis['summary']}")

                        issue_type = analysis["type"]
                        st.markdown(f"**Type:** `{issue_type}`")

                        st.markdown(render_priority(analysis["priority_score"]))

                        labels = analysis.get("suggested_labels", [])
                        if labels:
                            labels_md = " ".join(f"`{lbl}`" for lbl in labels)
                        else:
                            labels_md = "_None_"
                        st.markdown(f"**Suggested Labels:** {labels_md}")

                        st.markdown(f"**Potential Impact**  \n{analysis['potential_impact']}")

                        triage = derive_triage_suggestion(issue_type, analysis["priority_score"])
                        st.markdown("**Triage Suggestion (helper, not mandatory):**")
                        st.markdown(triage)

                        # ----- Raw JSON (optional) -----
                        if show_raw_json:
                            st.divider()
                            st.subheader("Raw JSON (LLM Output)")
                            raw_json_str = json.dumps(data["raw_llm_output"], indent=2)
                            st.code(raw_json_str, language="json")
                            st.download_button(
                                label="⬇️ Download JSON",
                                data=raw_json_str,
                                file_name=f"issue_{data['issue_number']}_analysis.json",
                                mime="application/json",
                            )

                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach backend at {BACKEND_URL}: {e}")


# -----------------------------------------------------------------------------
# TAB 2: HISTORY
# -----------------------------------------------------------------------------

with tab_history:
    st.header("Recent Analyses (this session)")

    if not st.session_state.history:
        st.write("No issues analyzed yet. Run something from the **Analyze Issue** tab first.")
    else:
        for idx, item in enumerate(st.session_state.history, start=1):
            with st.expander(
                f"{idx}. {item['repo_url']} – Issue #{item['issue_number']} – {item['issue_title'][:60]}..."
            ):
                st.markdown(f"**Repository:** `{item['repo_url']}`")
                st.markdown(f"**Issue #:** `{item['issue_number']}`")
                st.markdown(f"**Summary:** {item['ai_analysis']['summary']}")
                st.markdown(f"**Type:** `{item['ai_analysis']['type']}`")
                st.markdown(render_priority(item["ai_analysis"]["priority_score"]))

                # Extra: quickly load this issue back into the form
                if st.button(
                    f"Load this issue in Analyze tab",
                    key=f"load_{idx}",
                ):
                    # Set prefill values (safe) and rerun.
                    st.session_state.prefill_repo_url = item["repo_url"]
                    st.session_state.prefill_issue_number = item["issue_number"]
                    st.success("Loaded into form. Switch to the 'Analyze Issue' tab and click Analyze.")
                    st.rerun()


# -----------------------------------------------------------------------------
# TAB 3: ABOUT
# -----------------------------------------------------------------------------

with tab_about:
    st.header("About This App")

    st.markdown(
        """
        **Seedling Labs – AI-Powered GitHub Issue Assistant**

        This tool helps engineering teams quickly answer:

        > *“What is this issue about, and how important is it?”*

        **How it works:**

        - The **frontend** (this Streamlit app) collects a GitHub repo URL and issue number.
        - The **backend** (FastAPI) calls the GitHub REST API to fetch:
          - Issue title
          - Issue body
          - Issue comments
        - The backend sends this context to a **Hugging Face LLM** via the router API
          and asks for a specific JSON structure.

        The LLM returns JSON with:
        - `summary`
        - `type` (bug / feature_request / documentation / question / other)
        - `priority_score` (1–5 with justification)
        - `suggested_labels`
        - `potential_impact`

        The backend validates this JSON with Pydantic, then returns it here,
        where it is rendered into a clean, human-readable view with extra triage suggestions.

        The design is:
        - **Modular** – frontend, backend, and model are clearly separated.
        - **Robust** – handles missing comments, long bodies, disabled issues, etc.
        - **Extensible** – easy to add batch triage, persistence, or Slack/Teams integrations later.
        """
    )
