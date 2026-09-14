import os
import re
import time
from functools import lru_cache

import requests
from flask import Flask, request, render_template_string
from dotenv import load_dotenv
from markupsafe import escape
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is missing from .env")

# Gemini
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# Wikipedia
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

WIKIPEDIA_HEADERS = {
    "User-Agent": (
        "AI-Hallucination-Detector/1.0 "
        "(educational project; contact: local-project)"
    ),
    "Accept": "application/json",
}

# Small delay between Wikipedia requests
WIKI_REQUEST_DELAY = 0.8


# ============================================================
# GENERAL HELPERS
# ============================================================

STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "at",
    "for",
    "from",
    "by",
    "with",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "as",
    "than",
    "into",
    "over",
    "under",
    "around",
    "during",
    "through",
    "can",
    "could",
    "may",
    "might",
    "do",
    "does",
    "did",
    "not",
    "no",
    "yes",
    "very",
    "more",
    "less",
    "most",
    "some",
    "any",
    "all",
    "both",
    "their",
    "they",
    "we",
    "you",
    "he",
    "she",
    "human",
    "humans",
    "people",
}


# Pages that are usually unrelated to scientific fact verification
IRRELEVANT_CONTEXT_WORDS = {
    "film",
    "movie",
    "television",
    "tv",
    "series",
    "sitcom",
    "episode",
    "actor",
    "actress",
    "character",
    "fictional",
    "novel",
    "book",
    "comic",
    "comics",
    "album",
    "song",
    "band",
    "soundtrack",
    "game",
    "video",
    "musician",
    "singer",
    "producer",
    "director",
    "screenplay",
    "fiction",
    "cartoon",
}


def normalize_text(text):
    """Normalize text for comparison."""
    if not text:
        return ""

    text = text.lower()
    text = text.replace("°", " degree ")
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_keywords(text):
    """Extract meaningful keywords from a claim."""
    normalized = normalize_text(text)

    words = normalized.split()

    keywords = []

    for word in words:
        if word in STOP_WORDS:
            continue

        if len(word) <= 2:
            continue

        keywords.append(word)

    return list(dict.fromkeys(keywords))


def word_family(word):
    """
    Generate simple word-family variants.

    This helps connect:
    revolves -> revolution -> orbit
    freezing -> freeze -> ice
    respiration -> breathing
    """
    families = {
        "revolve": {
            "revolve",
            "revolves",
            "revolved",
            "revolving",
            "revolution",
            "revolutionary",
            "orbit",
            "orbits",
            "orbital",
            "orbiting",
        },
        "revolves": {
            "revolve",
            "revolves",
            "revolved",
            "revolving",
            "revolution",
            "revolutionary",
            "orbit",
            "orbits",
            "orbital",
            "orbiting",
        },
        "revolution": {
            "revolve",
            "revolves",
            "revolved",
            "revolving",
            "revolution",
            "orbit",
            "orbits",
            "orbital",
            "orbiting",
        },
        "orbit": {
            "orbit",
            "orbits",
            "orbital",
            "orbiting",
            "revolve",
            "revolves",
            "revolution",
        },
        "freezes": {
            "freeze",
            "freezes",
            "freezing",
            "frozen",
            "ice",
        },
        "freeze": {
            "freeze",
            "freezes",
            "freezing",
            "frozen",
            "ice",
        },
        "frozen": {
            "freeze",
            "freezes",
            "freezing",
            "frozen",
            "ice",
        },
        "respiration": {
            "respiration",
            "respire",
            "respiring",
            "breathing",
            "breathe",
            "oxygen",
        },
        "respire": {
            "respiration",
            "respire",
            "respiring",
            "breathing",
            "breathe",
            "oxygen",
        },
        "breathing": {
            "respiration",
            "respire",
            "respiring",
            "breathing",
            "breathe",
            "oxygen",
        },
        "larger": {
            "larger",
            "large",
            "size",
            "area",
        },
        "larger": {
            "larger",
            "large",
            "size",
            "area",
        },
    }

    return families.get(word, {word})


def expanded_keywords(keywords):
    """Expand keywords with related word families."""
    expanded = set()

    for keyword in keywords:
        expanded.update(word_family(keyword))

    return expanded


def text_contains_word(text, word):
    """Whole-word matching."""
    if not text or not word:
        return False

    return bool(
        re.search(
            rf"\b{re.escape(word)}\b",
            normalize_text(text),
        )
    )


def count_matches(text, keywords):
    """Count unique keyword/family matches in text."""
    normalized = normalize_text(text)

    count = 0
    matched = set()

    for keyword in keywords:
        variants = word_family(keyword)

        for variant in variants:
            if re.search(rf"\b{re.escape(variant)}\b", normalized):
                matched.add(keyword)
                break

    count = len(matched)

    return count, matched


# ============================================================
# HTTP REQUEST HELPER
# ============================================================

def wikipedia_request(params):
    """
    Wikipedia API request with retry handling.

    Handles:
    - 429 Too Many Requests
    - 500/502/503/504 server errors
    """

    for attempt in range(4):

        try:
            time.sleep(WIKI_REQUEST_DELAY)

            response = requests.get(
                WIKIPEDIA_API,
                params=params,
                headers=WIKIPEDIA_HEADERS,
                timeout=15,
            )

            if response.status_code == 429:
                wait_time = 2 ** attempt
                print(
                    f"Wikipedia rate limited. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue

            if response.status_code in {500, 502, 503, 504}:
                wait_time = 2 ** attempt
                print(
                    f"Wikipedia server error "
                    f"{response.status_code}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            return response.json()

        except requests.RequestException as error:

            print("Wikipedia Request Error:", error)

            if attempt < 3:
                time.sleep(2 ** attempt)

    return {}


# ============================================================
# WIKIPEDIA SEARCH
# ============================================================

@lru_cache(maxsize=128)
def wikipedia_search(query):
    """
    Search Wikipedia and return candidate pages.
    """

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": 0,
        "srsort": "relevance",
        "srwhat": "text",
        "srinfo": "totalhits|suggestion|rewrittenquery",
        "srprop": "snippet|titlesnippet",
        "srlimit": 20,
        "format": "json",
        "utf8": 1,
    }

    data = wikipedia_request(params)

    if not data:
        return []

    return data.get("query", {}).get("search", [])


# ============================================================
# STRONG WIKIPEDIA SOURCE FILTERING
# ============================================================

def is_irrelevant_page(title, snippet):
    """
    Reject pages that are clearly unrelated to scientific
    or factual verification.
    """

    combined = normalize_text(
        f"{title} {snippet}"
    )

    words = set(combined.split())

    # Strong entertainment-context rejection
    entertainment_hits = (
        words.intersection(IRRELEVANT_CONTEXT_WORDS)
    )

    # Special protection:
    # "3rd Rock from the Sun"
    if "3rd rock from the sun" in combined:
        return True

    # If several entertainment terms appear,
    # page is probably unrelated.
    if len(entertainment_hits) >= 2:
        return True

    return False


def score_wikipedia_result(title, snippet, claim_keywords):
    """
    Calculate a relevance score for a Wikipedia result.

    Strong title matches > snippet matches.
    Exact phrase gets a large bonus.
    Unrelated contexts receive heavy penalties.
    """

    normalized_title = normalize_text(title)
    normalized_snippet = normalize_text(snippet)

    score = 0

    title_matches = set()
    snippet_matches = set()

    # --------------------------------------------------------
    # Title matching
    # --------------------------------------------------------

    for keyword in claim_keywords:

        variants = word_family(keyword)

        for variant in variants:

            if re.search(
                rf"\b{re.escape(variant)}\b",
                normalized_title,
            ):
                title_matches.add(keyword)
                break

    # --------------------------------------------------------
    # Snippet matching
    # --------------------------------------------------------

    for keyword in claim_keywords:

        variants = word_family(keyword)

        for variant in variants:

            if re.search(
                rf"\b{re.escape(variant)}\b",
                normalized_snippet,
            ):
                snippet_matches.add(keyword)
                break

    # --------------------------------------------------------
    # Scoring
    # --------------------------------------------------------

    score += len(title_matches) * 14
    score += len(snippet_matches) * 3

    # Exact multi-word phrase bonus
    claim_phrase = " ".join(claim_keywords)

    if (
        claim_phrase
        and claim_phrase in normalized_title
    ):
        score += 30

    if (
        claim_phrase
        and claim_phrase in normalized_snippet
    ):
        score += 20

    # --------------------------------------------------------
    # Relevance penalties
    # --------------------------------------------------------

    if is_irrelevant_page(title, snippet):
        score -= 80

    # Flat Earth is not a reliable direct source for
    # ordinary astronomy claims.
    if "flat earth" in normalized_title:
        score -= 70

    # Counter-Earth is usually irrelevant to normal
    # Earth/Sun orbital claims.
    if "counter earth" in normalized_title:
        score -= 70

    # Special protection against "3rd Rock from the Sun"
    if (
        "sun" in claim_keywords
        and "3rd rock from the sun" in normalized_title
    ):
        score -= 100

    # If title has no meaningful claim concept,
    # don't allow snippet-only weak matches too easily.
    if len(title_matches) == 0:
        score -= 10

    # Very weak one-word result
    if (
        len(claim_keywords) >= 3
        and len(title_matches) == 0
        and len(snippet_matches) <= 1
    ):
        score -= 25

    return {
        "score": score,
        "title_matches": title_matches,
        "snippet_matches": snippet_matches,
    }


def get_relevant_wikipedia_pages(claim):
    """
    Find only strongly relevant Wikipedia pages.
    """

    claim_keywords = extract_keywords(claim)

    if not claim_keywords:
        return []

    search_query = " ".join(claim_keywords)

    results = wikipedia_search(search_query)

    scored_results = []

    for result in results:

        title = result.get("title", "").strip()

        snippet = result.get("snippet", "").strip()

        if not title:
            continue

        scoring = score_wikipedia_result(
            title,
            snippet,
            claim_keywords,
        )

        score = scoring["score"]

        title_matches = scoring["title_matches"]
        snippet_matches = scoring["snippet_matches"]

        total_matches = (
            len(title_matches)
            + len(snippet_matches)
        )

        # ----------------------------------------------------
        # Minimum relevance rules
        # ----------------------------------------------------

        keep = False

        # Strong title relevance
        if len(title_matches) >= 2 and score >= 12:
            keep = True

        # One title match + supporting snippet
        elif (
            len(title_matches) >= 1
            and len(snippet_matches) >= 2
            and score >= 12
        ):
            keep = True

        # Exact strong match
        elif score >= 30:
            keep = True

        # For short/simple claims
        elif (
            len(claim_keywords) <= 2
            and total_matches >= 2
            and score >= 12
        ):
            keep = True

        if not keep:
            continue

        scored_results.append(
            {
                "title": title,
                "snippet": snippet,
                "score": score,
            }
        )

    # Highest relevance first
    scored_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Remove duplicate titles
    # --------------------------------------------------------

    unique_results = []

    seen_titles = set()

    for result in scored_results:

        normalized_title = normalize_text(
            result["title"]
        )

        if normalized_title in seen_titles:
            continue

        seen_titles.add(normalized_title)

        unique_results.append(result)

    # Return only top 3 strongest sources
    return unique_results[:3]


# ============================================================
# WIKIPEDIA EVIDENCE
# ============================================================

@lru_cache(maxsize=128)
def get_wikipedia_evidence(title):
    """
    Fetch the introduction/extract of a Wikipedia page.
    """

    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "titles": title,
        "format": "json",
        "utf8": 1,
    }

    data = wikipedia_request(params)

    if not data:
        return ""

    pages = (
        data.get("query", {})
        .get("pages", {})
    )

    for page in pages.values():

        extract = page.get("extract", "")

        if extract:
            return extract[:1800]

    return ""


def build_wikipedia_evidence(claim):
    """
    Build evidence from strongly relevant sources only.
    """

    pages = get_relevant_wikipedia_pages(claim)

    evidence_items = []
    source_list = []

    for page in pages:

        title = page["title"]

        evidence = get_wikipedia_evidence(title)

        if not evidence:
            continue

        # ----------------------------------------------------
        # Final evidence relevance check
        # ----------------------------------------------------

        claim_keywords = extract_keywords(claim)

        evidence_matches, _ = count_matches(
            evidence,
            claim_keywords,
        )

        # Require evidence itself to have meaningful overlap.
        if len(claim_keywords) >= 3:
            if evidence_matches < 2:
                continue
        else:
            if evidence_matches < 1:
                continue

        evidence_items.append(
            {
                "title": title,
                "text": evidence,
            }
        )

        source_url = (
            "https://en.wikipedia.org/wiki/"
            + title.replace(" ", "_")
        )

        source_list.append(
            {
                "title": title,
                "url": source_url,
            }
        )

    return evidence_items, source_list


# ============================================================
# GEMINI VERIFICATION
# ============================================================

def verify_with_gemini(claim, evidence_items):
    """
    Verify the claim using Gemini and Wikipedia evidence.
    """

    if not client:
        return ""

    evidence_text = ""

    if evidence_items:

        for item in evidence_items[:3]:

            evidence_text += (
                f"\nSOURCE: {item['title']}\n"
                f"EVIDENCE: {item['text'][:1200]}\n"
            )

    else:

        evidence_text = (
            "\nNo reliable Wikipedia evidence "
            "was found.\n"
        )

    prompt = f"""
You are a careful fact verification assistant.

Analyze the following claim:

CLAIM:
{claim}

Use the provided Wikipedia evidence when it is relevant.

WIKIPEDIA EVIDENCE:
{evidence_text}

Decide whether the claim is:

TRUE
FALSE
UNCERTAIN

Rules:

1. TRUE means the claim is supported by reliable evidence.
2. FALSE means the claim contradicts reliable evidence.
3. UNCERTAIN means there is not enough reliable evidence.
4. Do not invent sources.
5. Do not invent URLs.
6. If the claim is scientifically established and the evidence supports it, prefer TRUE.
7. If the claim directly contradicts established scientific evidence, prefer FALSE.
8. Keep the explanation concise.
9. The verdict must be exactly TRUE, FALSE, or UNCERTAIN.

Return exactly this format:

VERDICT: TRUE
EXPLANATION: Short explanation.
"""

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if response and response.text:
            return response.text.strip()

    except Exception as error:

        print("Gemini Error:", error)

    return ""


# ============================================================
# LLAMA 3.2 VERIFICATION
# ============================================================

def verify_with_ollama(claim):
    """
    Verify claim using local Llama 3.2 through Ollama.
    """

    prompt = f"""
You are a fact verification assistant.

Analyze this claim:

CLAIM:
{claim}

Decide whether it is:

TRUE
FALSE
UNCERTAIN

Return exactly:

VERDICT: TRUE
EXPLANATION: Short explanation.

The verdict must be exactly TRUE, FALSE, or UNCERTAIN.

Rules:
- Use general scientific and factual knowledge.
- Do not invent sources.
- Do not invent URLs.
- Keep the explanation concise.
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("response", "").strip()

    except Exception as error:

        print("Ollama Error:", error)

    return ""


# ============================================================
# RESPONSE PARSING
# ============================================================

def parse_response(text):
    """
    Parse AI response safely.
    """

    if not text:
        return {
            "verdict": "UNCERTAIN",
            "explanation": "No verification response was received.",
        }

    verdict_match = re.search(
        r"VERDICT:\s*(TRUE|FALSE|UNCERTAIN)",
        text,
        re.IGNORECASE,
    )

    explanation_match = re.search(
        r"EXPLANATION:\s*(.*?)(?=\n\s*EVIDENCE:|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if verdict_match:

        verdict = verdict_match.group(1).upper()

    else:

        upper_text = text.upper()

        if "FALSE" in upper_text:
            verdict = "FALSE"

        elif "TRUE" in upper_text:
            verdict = "TRUE"

        else:
            verdict = "UNCERTAIN"

    if explanation_match:

        explanation = (
            explanation_match.group(1)
            .strip()
        )

    else:

        explanation = text.strip()

        # Remove accidental verdict line
        explanation = re.sub(
            r"VERDICT:\s*(TRUE|FALSE|UNCERTAIN)",
            "",
            explanation,
            flags=re.IGNORECASE,
        ).strip()

    return {
        "verdict": verdict,
        "explanation": explanation,
    }


# ============================================================
# COMBINE AI VERDICTS
# ============================================================

def combine_verdicts(gemini_verdict, llama_verdict):
    """
    Combine Gemini and Llama decisions.

    Both agree -> use their decision.
    Disagreement -> UNCERTAIN.
    """

    if (
        gemini_verdict in {"TRUE", "FALSE"}
        and llama_verdict in {"TRUE", "FALSE"}
    ):

        if gemini_verdict == llama_verdict:
            return gemini_verdict

        return "UNCERTAIN"

    # If only Gemini is available
    if gemini_verdict in {"TRUE", "FALSE"}:
        return gemini_verdict

    # If only Llama is available
    if llama_verdict in {"TRUE", "FALSE"}:
        return llama_verdict

    return "UNCERTAIN"


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    final_verdict,
    gemini_verdict,
    llama_verdict,
    evidence_items,
):
    """
    Calculate a simple confidence score.

    This is an application confidence indicator,
    not a statistical probability.
    """

    confidence = 50

    # Both models agree
    if (
        gemini_verdict in {"TRUE", "FALSE"}
        and llama_verdict in {"TRUE", "FALSE"}
        and gemini_verdict == llama_verdict
    ):
        confidence = 90

    # Only one model available
    elif (
        gemini_verdict in {"TRUE", "FALSE"}
        or llama_verdict in {"TRUE", "FALSE"}
    ):
        confidence = 70

    # Strong evidence
    if len(evidence_items) >= 2:
        confidence += 5

    elif len(evidence_items) == 1:
        confidence += 3

    # Keep between 0 and 100
    confidence = max(
        0,
        min(100, confidence),
    )

    # Agreement + evidence
    if (
        gemini_verdict == llama_verdict
        and final_verdict in {"TRUE", "FALSE"}
        and len(evidence_items) >= 1
    ):
        confidence = 100

    return confidence


# ============================================================
# FINAL DETECTION
# ============================================================

def detect_hallucination(claim):
    """
    Main verification pipeline.
    """

    claim = claim.strip()

    # --------------------------------------------------------
    # Wikipedia
    # --------------------------------------------------------

    evidence_items, sources = (
        build_wikipedia_evidence(claim)
    )

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    gemini_raw = verify_with_gemini(
        claim,
        evidence_items,
    )

    gemini_result = parse_response(
        gemini_raw
    )

    # --------------------------------------------------------
    # Llama
    # --------------------------------------------------------

    llama_raw = verify_with_ollama(
        claim
    )

    llama_result = parse_response(
        llama_raw
    )

    # --------------------------------------------------------
    # Combined verdict
    # --------------------------------------------------------

    final_verdict = combine_verdicts(
        gemini_result["verdict"],
        llama_result["verdict"],
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = calculate_confidence(
        final_verdict,
        gemini_result["verdict"],
        llama_result["verdict"],
        evidence_items,
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    explanation_parts = []

    if gemini_result["explanation"]:
        explanation_parts.append(
            f"Gemini: {gemini_result['explanation']}"
        )

    if llama_result["explanation"]:
        explanation_parts.append(
            f"Llama 3.2: {llama_result['explanation']}"
        )

    if (
        gemini_result["verdict"]
        == llama_result["verdict"]
        and gemini_result["verdict"]
        in {"TRUE", "FALSE"}
    ):

        explanation_parts.append(
            "Final decision: Both AI models agree."
        )

    elif (
        gemini_result["verdict"]
        in {"TRUE", "FALSE"}
        and llama_result["verdict"]
        in {"TRUE", "FALSE"}
    ):

        explanation_parts.append(
            "Final decision: The AI models disagree, "
            "so the result is marked UNCERTAIN."
        )

    explanation = " ".join(
        explanation_parts
    )

    # --------------------------------------------------------
    # Evidence text
    # --------------------------------------------------------

    evidence_display = []

    for item in evidence_items[:3]:

        short_text = item["text"].strip()

        if len(short_text) > 450:
            short_text = (
                short_text[:450].rsplit(" ", 1)[0]
                + "..."
            )

        evidence_display.append(
            {
                "title": item["title"],
                "text": short_text,
            }
        )

    return {
        "claim": claim,
        "final_verdict": final_verdict,
        "confidence": confidence,
        "gemini_verdict": gemini_result["verdict"],
        "llama_verdict": llama_result["verdict"],
        "explanation": explanation,
        "evidence": evidence_display,
        "sources": sources,
    }


# ============================================================
# HTML TEMPLATE
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>AI Hallucination Detector</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    color: #222;
}

.container {
    max-width: 950px;
    margin: 40px auto;
    padding: 20px;
}

.card {
    background: white;
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 20px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
}

h1 {
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}

textarea {
    width: 100%;
    min-height: 140px;
    padding: 15px;
    border: 1px solid #ccc;
    border-radius: 10px;
    font-size: 16px;
    resize: vertical;
}

button {
    width: 100%;
    margin-top: 15px;
    padding: 14px;
    border: none;
    border-radius: 10px;
    background: #222;
    color: white;
    font-size: 17px;
    cursor: pointer;
}

button:hover {
    opacity: 0.9;
}

.result-title {
    margin-top: 0;
}

.verdict {
    font-size: 24px;
    font-weight: bold;
    margin: 15px 0;
}

.true {
    color: green;
}

.false {
    color: red;
}

.uncertain {
    color: orange;
}

.section {
    margin-top: 25px;
}

.section h3 {
    margin-bottom: 10px;
}

.source {
    display: block;
    margin: 8px 0;
    color: #1565c0;
    text-decoration: none;
    word-break: break-word;
}

.source:hover {
    text-decoration: underline;
}

.evidence-box {
    background: #f7f7f7;
    border-left: 4px solid #777;
    padding: 12px;
    margin: 10px 0;
    border-radius: 6px;
}

.model {
    padding: 10px;
    background: #fafafa;
    border-radius: 8px;
    margin: 8px 0;
}

.small {
    color: #666;
    font-size: 14px;
}

.error {
    color: red;
    font-weight: bold;
}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h1>🤖 AI Hallucination Detector</h1>

<div class="subtitle">
Gemini + Llama 3.2 + Wikipedia Evidence
</div>

<form method="POST">

<textarea
    name="claim"
    placeholder="Enter a claim to verify..."
    required
>{{ claim }}</textarea>

<button type="submit">
🔍 Verify Claim
</button>

</form>

</div>


{% if result %}

<div class="card">

<h2 class="result-title">
📊 Verification Results
</h2>

<p>
📚 <strong>Verification mode:</strong>
Gemini AI + Llama 3.2 + Wikipedia evidence.
</p>

<p>
🤖 Two AI models are used for verification.
</p>

<p>
🔗 Sources below are clickable reference pages.
</p>


<div class="section">

<h3>📌 Claim</h3>

<p>
<strong>{{ result.claim }}</strong>
</p>

</div>


<div class="section">

<h3>🔍 AI Hallucination Detection Result</h3>

{% if result.final_verdict == "TRUE" %}

<div class="verdict true">
🟢 LIKELY TRUE
</div>

{% elif result.final_verdict == "FALSE" %}

<div class="verdict false">
🔴 LIKELY FALSE / HALLUCINATION
</div>

{% else %}

<div class="verdict uncertain">
🟠 UNCERTAIN
</div>

{% endif %}

<p>
📊 <strong>Confidence:</strong>
{{ result.confidence }}%
</p>

</div>


<div class="section">

<h3>🤖 AI Model Comparison</h3>

<div class="model">
🧠 <strong>Gemini:</strong>
{{ result.gemini_verdict }}
</div>

<div class="model">
🦙 <strong>Llama 3.2:</strong>
{{ result.llama_verdict }}
</div>

</div>


<div class="section">

<h3>💡 Explanation</h3>

<p>
{{ result.explanation }}
</p>

</div>


<div class="section">

<h3>📚 Evidence</h3>

{% if result.evidence %}

{% for item in result.evidence %}

<div class="evidence-box">

<strong>
{{ item.text }}
</strong>

<p class="small">
Source: {{ item.title }}
</p>

</div>

{% endfor %}

{% else %}

<p class="small">
No strongly relevant Wikipedia evidence was found.
</p>

{% endif %}

</div>


<div class="section">

<h3>🔗 Sources</h3>

{% if result.sources %}

{% for source in result.sources %}

<a
    class="source"
    href="{{ source.url }}"
    target="_blank"
    rel="noopener noreferrer"
>
🔗 <strong>{{ source.title }}</strong>
</a>

{% endfor %}

{% else %}

<p class="small">
No strongly relevant Wikipedia sources found.
</p>

{% endif %}

</div>

</div>

{% endif %}

</div>

</body>

</html>
"""


# ============================================================
# FLASK ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    claim = ""
    result = None
    error = None

    if request.method == "POST":

        claim = request.form.get(
            "claim",
            "",
        ).strip()

        if not claim:

            error = "Please enter a claim."

        elif len(claim) > 1000:

            error = (
                "Claim is too long. "
                "Please keep it under 1000 characters."
            )

        else:

            try:

                result = detect_hallucination(
                    claim
                )

            except Exception as exc:

                print(
                    "Application Error:",
                    exc,
                )

                error = (
                    "Something went wrong while "
                    "verifying the claim."
                )

    if error:

        return render_template_string(
            HTML_TEMPLATE
            + """
            <div class="card">
                <p class="error">
                    {{ error }}
                </p>
            </div>
            """,
            claim=claim,
            result=result,
            error=error,
        )

    return render_template_string(
        HTML_TEMPLATE,
        claim=claim,
        result=result,
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI Hallucination Detector")
    print("=" * 60)

    print("Gemini:", GEMINI_MODEL)
    print("Ollama:", OLLAMA_MODEL)
    print("Wikipedia: Enabled")

    print(
        "Open browser at: "
        "http://127.0.0.1:5000"
    )

    print("=" * 60)

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
    )