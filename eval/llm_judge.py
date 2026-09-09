"""LLM-as-judge scoring for rule-interpretation answers. Claude scores a
candidate answer against the reference answer and the source rulebook passage,
1-5, with a rationale -- not just "did it match" but "is it actually correct
per the rule text."
"""
import json

from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"  # cost-conscious: hundreds of judge calls across 4 systems x held-out set

JUDGE_SYSTEM_PROMPT = """You are grading an AI system's answer to an NFL \
rulebook question. You will see the question, the source rule passage the \
reference answer was written from, the reference answer, and the candidate \
answer to grade.

Score the candidate 1-5:
5 = fully correct, matches the rule's substance, no material omission
4 = correct but missing a minor detail
3 = partially correct or vague enough to be misleading
2 = mostly wrong but touches the right topic
1 = wrong or contradicts the rule

Judge the candidate against what the rule actually says, not just similarity \
to the reference wording -- a differently-worded but substantively correct \
answer should score 5.

Return strict JSON: {"score": <int 1-5>, "rationale": "<one sentence>"}. No \
prose outside the JSON."""


def judge(client: Anthropic, question: str, source_text: str,
          reference_answer: str, candidate_answer: str) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Source rule passage: {source_text}\n\n"
                f"Reference answer: {reference_answer}\n\n"
                f"Candidate answer: {candidate_answer}"
            ),
        }],
    )
    raw = next(block.text for block in resp.content if block.type == "text")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        return json.JSONDecoder().raw_decode(raw[start:])[0]
