import json


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
    if text.endswith("```"):
        text = text[:text.rindex("```")]
    return text.strip()


def parse_codings(text: str) -> list[dict]:
    """Parse LLM response text (fences already stripped) into a list of coding dicts.

    Handles: double-brace bug, raw_response wrapper.
    """
    if text.startswith("{{"):
        text = text[1:]
    raw = json.loads(text)
    if isinstance(raw, dict) and "raw_response" in raw and "codings" not in raw:
        raw = json.loads(strip_fences(raw["raw_response"]))
    return raw if isinstance(raw, list) else raw.get("codings", [])
