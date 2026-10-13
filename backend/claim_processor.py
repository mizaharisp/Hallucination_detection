
import spacy
import re

nlp = spacy.load("en_core_web_sm")


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
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return text


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

            subject_text = doc[
                subject.left_edge.i:subject.right_edge.i + 1
            ].text

            first_claim = sentence

            for i, word in enumerate(doc):
                if word.text.lower() == "and":
                    first_claim = doc[:i].text.strip()
                    break

            second_phrase = doc[
                token.left_edge.i:token.right_edge.i + 1
            ].text.strip()

            second_claim = subject_text + " " + second_phrase

            return [first_claim, second_claim]

    return [sentence]


def process_gemini_response(gemini_response):
    if not isinstance(gemini_response, str) or not gemini_response.strip():
        return []

    text = gemini_response.replace("\r\n", "\n")
    text = fix_known_spacing(text)

    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

       
        if line.startswith("#") or re.fullmatch(r"[-*_]{3,}", line):
            continue

       
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)

        
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)

       
        if re.match(
            r"^(how AI works|key concepts|the types of AI|"
            r"common real-world examples|opportunities and challenges)$",
            line.rstrip(":").strip(),
            re.IGNORECASE,
        ):
            continue

        cleaned_lines.append(line)

    all_claims = []

    for line in cleaned_lines:
        doc = nlp(line)

        for sentence in doc.sents:
            claim = sentence.text.strip()

            if is_candidate_claim(claim):
             
                all_claims.extend(split_compound_claim(claim))

  
    return list(dict.fromkeys(
        claim for claim in all_claims if claim.strip()
    ))
