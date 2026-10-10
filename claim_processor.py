import spacy
import json
import requests
import re


def fix_known_spacing(text):
    replacements = {
        r"\bproblemson\b": "problems on",
        r"\bsingleaction\b": "single action",
        r"\bunderstand ing\b": "understanding",
        r"\bMachine Learning:\s*A\b": "Machine Learning: A",
        r"\bHere are(?=\d)": "Here are ",
        r"\bthem in\b": "them in",
        r"\bThe Three Stages ofAI\b": "The Three Stages of AI",
        r"\bEnablingcomputers\b": "Enabling computers",
        r"\bChatGPTseems\b": "ChatGPT seems",
        r"\bdoes notyet\b": "does not yet",
        r"\bhypotheticalAI\b": "hypothetical AI",
        r"\bincludingcreativity\b": "including creativity",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


nlp = spacy.load("en_core_web_sm")

def is_candidate_claim(sentence):
    sentence = sentence.strip()

    if sentence.endswith("?"):
        return False

    opinion_phrases = [
        "i think",
        "i believe",
        "i feel",
        "in my opinion",
        "i would say",
        "it seems to me",
        "personally"
    ]

    lower_sentence = sentence.lower()

    for phrase in opinion_phrases:
        if lower_sentence.startswith(phrase):
            return False

    return len(sentence.split()) > 3

def split_compound_claim(sentence):
    doc = nlp(sentence)

    for token in doc:

        if token.dep_ == "conj" and token.pos_ == "VERB":

            main_verb = token.head

            subject = None

            for child in main_verb.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subject = child
                    break

            if subject is None:
                return [sentence]

            subject_start = subject.left_edge.i
            subject_end = subject.right_edge.i

            subject_text = doc[
                subject_start:subject_end + 1
            ].text

            first_claim = sentence

            for i, word in enumerate(doc):
                if word.text.lower() == "and":
                    first_claim = doc[:i].text.strip()
                    break

            second_start = token.left_edge.i
            second_end = token.right_edge.i

            second_phrase = doc[
                second_start:second_end + 1
            ].text.strip()

            second_claim = subject_text + " " + second_phrase

            return [
                first_claim,
                second_claim
            ]

    return [sentence]



def process_gemini_response(gemini_response):
    text = gemini_response.replace("\r\n", "\n")
    text = fix_known_spacing(text)
    cleaned_lines = []
    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        # Skip Markdown headings and horizontal rules
        if line.startswith("#") or re.fullmatch(r"[-*_]{3,}", line):
            continue

        # Remove bullet and numbered-list markers
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)

        # Remove bold, italic and inline-code markers
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)

        # Skip common standalone section labels
        if re.match(
            r"^(how AI works|key concepts|the types of AI|"
            r"common real-world examples|opportunities and challenges)$",
            line.rstrip(":").strip(),
            re.IGNORECASE,
        ):
            continue

        cleaned_lines.append(line)

    # Split each remaining line into sentences
    all_claims = []

    for line in cleaned_lines:
        doc = nlp(line)

        for sentence in doc.sents:
            claim = sentence.text.strip()

            if is_candidate_claim(claim):
                all_claims.append(claim)

    # Remove exact duplicates while preserving order
    return list(dict.fromkeys(all_claims))

# Get a question from the user
query = input("Enter your question: ").strip()

if not query:
    print("Please enter a question.")
    raise SystemExit

# Connect to Member 1's Gemini backend
try:
    response = requests.post(
        "http://127.0.0.1:5000/generate",
        json={"prompt": query},
        timeout=120
    )

    response.raise_for_status()
    member1_data = response.json()

    gemini_response = member1_data.get("gemini_response")
    print("\n--- RAW GEMINI RESPONSE ---\n")
    print(repr(gemini_response))
    print("\n--- END RAW RESPONSE ---\n")

    if not gemini_response:
        print("Expected 'gemini_response' was not found.")
        print("Backend returned:", member1_data)
        raise SystemExit

except requests.RequestException as error:
    print("Could not connect to Member 1's backend.")
    print("Error:", error)
    raise SystemExit

# Process the real Gemini response
all_claims = process_gemini_response(gemini_response)

# Prepare JSON output for Member 3
claim_data = {
    "query": query,
    "claims": [
        {
            "claim_id": f"C{i}",
            "claim": claim
        }
        for i, claim in enumerate(all_claims, start=1)
    ]
}


print(json.dumps(claim_data, indent=2))

with open("claims_output.json", "w", encoding="utf-8") as f:
    json.dump(claim_data, f, indent=2, ensure_ascii=False)

print("Claims saved to claims_output.json")

