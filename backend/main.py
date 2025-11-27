"""
Seedling Labs – AI-Powered GitHub Issue Assistant (Free APIs)

This backend does 4 main things:

1. Accepts input from the frontend:
   - GitHub repository URL (public)
   - Issue number

2. Uses the GitHub API to fetch:
   - Issue title
   - Issue body (description)
   - Issue comments

3. Sends this information to a FREE Large Language Model (LLM)
   hosted on Hugging Face, via their "router" API using the
   OpenAI-compatible client.

4. Returns a CLEAN, STRUCTURED JSON with the following keys:
   {
     "summary": "one-sentence summary of the issue",
     "type": "bug | feature_request | documentation | question | other",
     "priority_score": "score 1-5 plus justification",
     "suggested_labels": ["label1", "label2", ...],
     "potential_impact": "impact sentence"
   }

The goal is to keep the code:
- Very readable (even for beginners)
- Well-commented
- Easy to maintain
"""

import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, HttpUrl, ValidationError

from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
env_file = BASE_DIR / ".env"

print("Looking for .env at:", env_file)

load_dotenv(dotenv_path=env_file, override=True)

HF_TOKEN = os.getenv("HF_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

print("HF token loaded:", HF_TOKEN is not None)
print("GH token loaded:", GITHUB_TOKEN is not None)
print("GH token prefix:", GITHUB_TOKEN[:7] if GITHUB_TOKEN else None)



# -----------------------------------------------------------------------------
# 1. CONFIGURATION (MODEL + TOKENS)
# -----------------------------------------------------------------------------

# Hugging Face model name.
# ":hf-inference" suffix tells Hugging Face we want to use their Inference API.
HF_MODEL = os.getenv(
    "HF_MODEL_NAME",
    "HuggingFaceTB/SmolLM3-3B:hf-inference",  # small, free-ish instruct model
)

# Hugging Face token (REQUIRED). Get it from:
# https://huggingface.co/settings/access-tokens
HF_TOKEN = os.getenv("HF_TOKEN")

# GitHub token (OPTIONAL). This is only to increase rate limits.
# If you don't set it, GitHub will still work for public repos, but with 60 req/hour limit.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not HF_TOKEN:
    # We fail fast so the user immediately knows something is wrong with setup.
    raise RuntimeError(
        "HF_TOKEN is not set. Please create a Hugging Face access token "
        "and set HF_TOKEN in your environment."
    )

# Create a client that talks to Hugging Face router but using OpenAI-style API.
# IMPORTANT: This is NOT the real OpenAI API; it just uses the same Python client.
hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",  # Hugging Face router endpoint
    api_key=HF_TOKEN,
)


# -----------------------------------------------------------------------------
# 2. DATA MODELS (REQUEST & RESPONSE SCHEMAS)
# -----------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """
    Incoming request from frontend.

    Example JSON:
    {
      "repo_url": "https://github.com/facebook/react",
      "issue_number": 12345
    }
    """
    repo_url: HttpUrl   # Validates that the input is a proper URL
    issue_number: int   # Issue number must be positive integer


class AIAnalysis(BaseModel):
    """
    AI-generated JSON schema EXACTLY as required in the assignment.

    {
      "summary": "A one-sentence summary of the user's problem or request.",
      "type": "bug | feature_request | documentation | question | other",
      "priority_score": "A score from 1 (low) to 5 (critical), with a brief justification.",
      "suggested_labels": ["label1", "label2", ...],
      "potential_impact": "A short sentence about impact on users."
    }
    """
    summary: str
    type: Literal["bug", "feature_request", "documentation", "question", "other"]
    priority_score: str
    suggested_labels: List[str]
    potential_impact: str


class AnalyzeResponse(BaseModel):
    """
    Full response we send back to the frontend.

    It contains:
    - Basic issue metadata
    - The AIAnalysis object
    - Raw LLM output (for debugging / showing JSON)
    """
    repo_url: str
    issue_number: int
    issue_title: str
    issue_html_url: Optional[str]
    ai_analysis: AIAnalysis
    raw_llm_output: Dict[str, Any]


# -----------------------------------------------------------------------------
# 3. FASTAPI APP INITIALIZATION + CORS
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Seedling Labs – GitHub Issue Assistant (Free APIs)",
    description=(
        "Fetches GitHub issue + comments and asks a free Hugging Face model "
        "to produce structured analysis JSON."
    ),
)

# CORS allows our frontend (Streamlit) to call this backend from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # for local dev; in production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# 4. HELPER FUNCTIONS – GITHUB URL PARSING & TRUNCATION
# -----------------------------------------------------------------------------

def parse_github_repo(repo_url: str) -> Tuple[str, str]:
    """
    Given a full repo URL like:
      https://github.com/facebook/react
    it returns:
      ("facebook", "react")

    It also handles:
      - trailing slashes
      - query params, e.g. ?tab=issues
    """
    if "github.com" not in repo_url:
        raise ValueError("URL does not appear to be a GitHub URL.")

    # Remove any query params or fragments (stuff after ? or #)
    cleaned = re.sub(r"[?#].*$", "", repo_url).rstrip("/")

    parts = cleaned.split("/")
    try:
        owner = parts[-2]
        repo = parts[-1]
    except IndexError:
        raise ValueError("Could not parse owner/repo from GitHub URL.")

    if not owner or not repo:
        raise ValueError("Owner or repo name is empty in the URL.")

    return owner, repo


def truncate_text(text: str, max_chars: int = 6000) -> str:
    """
    If the issue body or comments are VERY long, we cut them down.

    This avoids:
    - Blowing up the LLM context window
    - Slowing down the request too much

    We keep a "truncated" note at the end to tell the model content was cut.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    suffix = f"\n\n[Truncated: original length {len(text)} characters]"
    return text[: max_chars - len(suffix)] + suffix


# -----------------------------------------------------------------------------
# 5. GITHUB CLIENT – FETCH ISSUE + COMMENTS
# -----------------------------------------------------------------------------

def github_headers() -> Dict[str, str]:
    """
    Headers for GitHub API calls.

    If GITHUB_TOKEN is set, we include it to get higher rate limits.
    """
    headers = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


@lru_cache(maxsize=256)
def fetch_issue_and_comments(owner: str, repo: str, issue_number: int) -> Dict[str, Any]:
    """
    Fetch GitHub issue details and comments.

    - Uses GitHub REST API.
    - Handles common errors: 404 (issue not found), 410 (issues disabled).
    - Uses lru_cache so repeated calls for the same issue are fast.
    """
    base_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"

    # 1) Fetch main issue object
    response = requests.get(base_url, headers=github_headers(), timeout=15)

    if response.status_code == 404:
        # Issue number doesn't exist
        raise HTTPException(status_code=404, detail="Issue not found. Check repo URL and issue number.")

    if response.status_code == 410:
        # Some repos disable GitHub Issues (e.g., torvalds/linux)
        raise HTTPException(
            status_code=410,
            detail="Issues are disabled for this repository. Please try another repo.",
        )

    if not response.ok:
        # Any other error from GitHub (rate limit, server error, etc.)
        raise HTTPException(
            status_code=response.status_code,
            detail=f"GitHub API error: {response.text}",
        )

    issue_data = response.json()
    title = issue_data.get("title") or ""
    body = issue_data.get("body") or ""
    html_url = issue_data.get("html_url")

    # 2) Fetch comments (if any)
    comments_url = issue_data.get("comments_url")
    comments_list: List[str] = []

    if comments_url:
        comments_resp = requests.get(comments_url, headers=github_headers(), timeout=15)
        # If comments request fails, we don't crash the whole flow.
        if comments_resp.ok:
            raw_comments = comments_resp.json() or []
            for c in raw_comments:
                user_login = (c.get("user") or {}).get("login") or "unknown"
                comment_body = c.get("body") or ""
                comments_list.append(f"{user_login}: {comment_body}")

    # Combine all comments into one big text block.
    comments_block = "\n".join(comments_list) if comments_list else "No comments available."
    comments_block = truncate_text(comments_block, max_chars=6000)

    return {
        "title": title,
        "body": truncate_text(body, max_chars=6000),
        "html_url": html_url,
        "comments_block": comments_block,
    }


# -----------------------------------------------------------------------------
# 6. LLM HELPERS – PROMPT BUILDING + CALLING HUGGING FACE
# -----------------------------------------------------------------------------

def build_prompt(issue: Dict[str, Any]) -> str:
    """
    Build a clear prompt for the LLM.

    We:
    - Explain the exact JSON format we need.
    - Show the issue title, body, and comments.
    - Remind model to output ONLY JSON.
    """
    title = issue.get("title", "").strip() or "(no title)"
    body = issue.get("body", "").strip() or "(no body provided)"
    comments_block = issue.get("comments_block", "No comments available.")

    prompt = f"""
You are an assistant helping a fast-moving engineering team quickly understand and prioritize GitHub issues.

You will be given:
- Issue title
- Issue body
- Issue comments (possibly truncated if very long)

You MUST output EXACTLY ONE JSON object with this schema:

{{
  "summary": "A one-sentence summary of the user's problem or request.",
  "type": "One of: bug, feature_request, documentation, question, or other.",
  "priority_score": "A string containing a score from 1 (low) to 5 (critical), followed by a brief justification. Example: '4 - High: breaks login for many users.'",
  "suggested_labels": ["2-3 relevant GitHub labels (e.g., 'bug', 'UI', 'login-flow')."],
  "potential_impact": "A brief sentence on the potential impact on users if the issue is a bug. If not a bug, explain why the impact is limited."
}}

IMPORTANT RULES:
- Output MUST be valid JSON.
- DO NOT include any text before or after the JSON.
- Use double quotes for all strings.
- "suggested_labels" MUST be an array of 2–3 short strings.
- "type" MUST be exactly one of: "bug", "feature_request", "documentation", "question", "other".
- Even if content is truncated, still infer the most reasonable classification and priority.

Now analyze this GitHub issue:

ISSUE TITLE:
{title}

ISSUE BODY:
{body}

ISSUE COMMENTS (format: author: comment):
{comments_block}
"""
    return prompt.strip()


def extract_json(text: str) -> Dict[str, Any]:
    """
    The model might sometimes add extra text accidentally.
    This function:
    - Finds the first '{' and the last '}' in the response
    - Takes everything in between as JSON
    - Tries to parse it

    If parsing fails, we raise an HTTPException (500).
    """
    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end <= start:
        # No JSON object at all
        raise HTTPException(status_code=500, detail="LLM did not return a JSON object.")

    json_str = text[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse LLM JSON output: {str(e)}",
        )


def call_llm(prompt: str) -> Dict[str, Any]:
    """
    Call the Hugging Face model via the OpenAI-style client.

    Steps:
    1. Send the prompt.
    2. Get the text response.
    3. Try to extract JSON.
    4. Return both parsed JSON and raw text.

    If the model does not follow instructions, we raise an error.
    """
    try:
        completion = hf_client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON following the requested schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,   # Lower temp for more deterministic JSON
            max_tokens=512,
        )
    except Exception as e:
        # Any error while talking to Hugging Face (network, auth, model not found...)
        raise HTTPException(status_code=502, detail=f"Error calling HuggingFace LLM: {e}")

    # Extract model-generated text
    raw_text = completion.choices[0].message.content

    # Try to parse JSON from that text
    parsed = extract_json(raw_text)

    # We return both, so frontend can show the raw JSON and we can validate using pydantic.
    return {"parsed": parsed, "raw_text": raw_text}


# -----------------------------------------------------------------------------
# 7. API ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    """
    Simple health check endpoint.
    Use it to quickly verify that the backend is running.
    """
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    MAIN ENDPOINT called by the frontend.

    Flow:
    1. Validate issue_number is positive.
    2. Parse GitHub repo URL -> (owner, repo).
    3. Fetch issue + comments from GitHub.
    4. Build prompt and call LLM.
    5. Validate LLM JSON against AIAnalysis schema.
    6. Return structured response.
    """
    if req.issue_number <= 0:
        raise HTTPException(status_code=400, detail="Issue number must be a positive integer.")

    # Parse repo URL
    try:
        owner, repo = parse_github_repo(str(req.repo_url))
    except ValueError as e:
        # Bad GitHub URL
        raise HTTPException(status_code=400, detail=str(e))

    # Fetch issue data and comments from GitHub
    issue = fetch_issue_and_comments(owner, repo, req.issue_number)

    # Build LLM prompt and call model
    llm_result = call_llm(build_prompt(issue))

    # Validate that the parsed JSON matches our AIAnalysis schema
    try:
        analysis = AIAnalysis(**llm_result["parsed"])
    except ValidationError as e:
        # LLM failed to follow schema exactly
        raise HTTPException(
            status_code=500,
            detail=f"LLM JSON did not match expected schema: {e}",
        )

    # Return full response for the frontend
    return AnalyzeResponse(
        repo_url=str(req.repo_url),
        issue_number=req.issue_number,
        issue_title=issue["title"],
        issue_html_url=issue["html_url"],
        ai_analysis=analysis,
        raw_llm_output=llm_result["parsed"],
    )
