import json
from datetime import datetime
import requests

OLLAMA_URL = "http://127.0.0.1:11434"

def get_timestamp() -> str:
    """
    Get the current timestamp as a string in the format YYYYMMDD_HHMMSS.

    Returns:
        The current timestamp as a string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def strip_fences(text: str) -> str:
    """
    Strip code fences (```) from the start and end of the text, if present.

    Args:
        text: The text to strip code fences from.

    Returns:
        The text with code fences removed.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
    if text.endswith("```"):
        text = text[:text.rindex("```")]
    return text.strip()


def parse_codings(text: str) -> list[dict]:
    """
    Parse LLM response text (fences already stripped) into a list of coding dicts.

    Handles: double-brace bug, raw_response wrapper.

    Args:
        text: The LLM response text with code fences already stripped.

    Returns:
        A list of coding dictionaries parsed from the response.
    """

    # Double-brace bug
    if text.startswith("{{"):
        text = text[1:]

    # Raw response wrapper handling
    raw = json.loads(text)

    # If the raw response is a dict with a "raw_response" key but no "codings" key, parse the raw response.
    if isinstance(raw, dict) and "raw_response" in raw and "codings" not in raw:
        raw = json.loads(strip_fences(raw["raw_response"]))

    # Return the parsed codings, either as a list or from the "codings" key if present.
    return raw if isinstance(raw, list) else raw.get("codings", [])

def ollama_get_loaded_models() -> set[str]:
    """
    Get a list of models currently loaded in Ollama.

    Returns:
        A set of model names that are currently loaded.
    """


    response = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10)
    response.raise_for_status()

    return {
        item.get("name") or item.get("model")
        for item in response.json().get("models", [])
        if item.get("name") or item.get("model")
    }