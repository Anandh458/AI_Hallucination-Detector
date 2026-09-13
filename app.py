from flask import Flask, render_template, request
import requests
import urllib.parse
import re

app = Flask(__name__)


def identify_claims(text):
    sentences = re.split(r"[.!?]+", text)

    claims = []

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence) > 10:
            claims.append(sentence)

    return claims


def search_evidence(claim):
    encoded_claim = urllib.parse.quote(claim)

    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query"
        "&list=search"
        f"&srsearch={encoded_claim}"
        "&format=json"
        "&srlimit=3"
    )

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "AI-Hallucination-Detector/1.0"
            },
            timeout=5
        )

        if response.status_code != 200:
            return []

        data = response.json()

        return data.get("query", {}).get("search", [])

    except requests.RequestException:
        return []


def verify_claim(claim):
    results = search_evidence(claim)

    if not results:
        return {
            "status": "UNCERTAIN",
            "message": "No reliable evidence was retrieved.",
            "source": "No source found"
        }

    source_title = results[0].get(
        "title",
        "Unknown source"
    )

    # Conservative approach:
    # Finding a source does NOT mean the claim is proven true.
    return {
        "status": "NEEDS REVIEW",
        "message": "Relevant evidence source found. Manual verification is recommended.",
        "source": source_title
    }


def search_trusted_source(claim):
    query = urllib.parse.quote_plus(claim)

    return f"https://www.google.com/search?q={query}"


def detect_hallucination(text):
    text_lower = text.lower()

    warning_words = [
        "always",
        "never",
        "100%",
        "guaranteed",
        "definitely",
        "certainly"
    ]

    found = [
        word for word in warning_words
        if word in text_lower
    ]

    if len(found) >= 2:
        score = 80
        level = "HIGH RISK"

    elif len(found) == 1:
        score = 50
        level = "MEDIUM RISK"

    else:
        score = 10
        level = "LOW RISK"

    return level, score, found


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        text = request.form.get("text", "")

        if not text.strip():

            result = "Please enter some text."

        else:

            level, score, words = detect_hallucination(text)

            claims = identify_claims(text)

            result = (
                f"{level} | "
                f"Confidence Score: {score}%"
            )

            if claims:

                result += (
                    "<br><br>"
                    "<b>Identified Claims:</b>"
                    "<br><br>"
                )

                for claim in claims:

                    verification = verify_claim(claim)

                    source = search_trusted_source(claim)

                    result += (
                        f"📌 Claim: {claim}<br>"
                    )

                    result += (
                        f"🔎 Verification: "
                        f"{verification['status']}<br>"
                    )

                    result += (
                        f"💡 {verification['message']}<br>"
                    )

                    result += (
                        f"📖 Source: "
                        f"{verification['source']}<br>"
                    )

                    result += (
                        f'📚 <a href="{source}" '
                        f'target="_blank">'
                        f'Search Evidence</a>'
                        f"<br><br>"
                    )

            if words:

                result += (
                    "<b>Warning words:</b> "
                    f"{', '.join(words)}"
                )

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)