# Interface with LLMs via litellm

import os
import sys
import time
import json
import requests
from typing import Any, cast

from rich import print
import litellm

from ethnocoder.utils import (
    strip_fences, get_timestamp,
    OLLAMA_URL,
)

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

def ollama_get_context_length(
        model: str, 
    ) -> int | None:
    """
    Get the context length for the specified Ollama model.

    Args:
        model (str): The model name.

    Returns:
        int | None: The context length if available, otherwise None.
    """

    # Call the Ollama API to get model information.
    r = requests.post(f"{OLLAMA_URL}/api/show", json={"model": model}, timeout=10)
    r.raise_for_status()

    # Parse the response JSON to extract model information.
    info = r.json().get("model_info", {})

    # Look for the context length key in the model information.
    for key, value in info.items():
        if key.endswith(".context_length"):
            return value

    # If no context length key is found, return None.
    return None

def get_max_context(
        model: str,
        # api_base: str | None = None,
) -> int:
    """
    Get the maximum context size for the specified model.

    Args:
        model (str): The model name.
        # api_base (str | None): The API base URL, if any.

    Returns:
        int: The maximum context size for the model.
    """


    if model.startswith("ollama/"):
        return ollama_get_context_length(model.replace("ollama/", ""))
    
    elif model.startswith("openai/"):

        # Ask litellm, then try the custom models.json file.
        max_context = None
        try:
            max_context = litellm.get_model_info(model)["max_input_tokens"]
        except litellm.exceptions.ModelNotMappedError:

            # Try models.json file.
            models_path = "models.json"
            if os.path.exists(models_path):
                with open(models_path, "r", encoding="utf-8") as f:
                    models_data = json.load(f)
                    if model in models_data:
                        max_context = models_data[model].get("context_window")

        if max_context is None:
            raise ValueError(f"Unable to determine max context for model {model}")

        return max_context
    
    elif model.startswith("lm_studio/"):
        raise NotImplementedError("get_max_context is not implemented yet.")
    
    else:
        raise NotImplementedError("get_max_context is not implemented yet.")

def llm_stream(
    messages: list[dict],
    model: str,
    is_anthropic: bool,
    api_base: str | None,
    context_budget: int,
    bool_log: bool = False,
    bool_thinking: bool = False,
) -> tuple[str, dict]:
    """
    Stream the LLM response for the given messages and return the final text.

    Args:
        messages: The list of messages to send to the LLM.
        model: The model to use for the LLM.
        is_anthropic: Whether the model is an Anthropic model.
        api_base: The base URL for the API, if any.
        context_budget: Token budget per request.
        bool_log: Whether to log the messages to a file.
        bool_thinking: Whether the LLM should display its thinking process.
        
    Returns:
        final_text: The final text response from the LLM.
        meta: Metadata about the LLM call, including elapsed time and model load status.
    """

    # Initialise. We will use litellm with the streaming option enabled.
    kwargs: dict = {"stream": True}

    # Add response_format option if using a non-Anthropic model that is not lm_studio.
    if not is_anthropic and not model.startswith("lm_studio/"):
        kwargs["response_format"] = {"type": "json_object"}

    # Set api base url if provided.
    if api_base:
        kwargs["api_base"] = api_base

    # If it's ollama, get the list of loaded models.
    if model.startswith("ollama/"):
        loaded_models = ollama_get_loaded_models()

        if model[len("ollama/"):] not in loaded_models:
            model_loaded = False
        else:
            model_loaded = True
    else:
        model_loaded = None

    # Log messages now
    if bool_log:

        os.makedirs("logs", exist_ok=True)

        # Output the exact messages to a timestamped log
        ts = get_timestamp()
        with open(f"logs/{ts}_messages.log", "w", encoding="utf-8") as f:
            for message in messages:
                f.write(json.dumps(message, ensure_ascii=False, indent=2))
                f.write("\n")

    # Estimate token usage now.
    prompt_tokens = litellm.token_counter(
        model=model,       # e.g. "ollama/qwen3.5:35b"
        messages=messages,
    )

    if prompt_tokens:
        print(f"[blue]Estimated prompt tokens: {prompt_tokens:,}[/blue]")

    # Send the messages to the LLM and stream the response.
    # If it's a remote call (i.e. openai/), each attempt will have a timeout.
    use_timeout = model.startswith("openai/")
    llm_provider = model.split("/")[0]
    timeout_s = 300
    max_attempts = 5 if use_timeout else 1

    # Prevent litellm from passing parameters to models that can't use them
    litellm.drop_params = True

    for attempt in range(1, max_attempts + 1):
        
        # Initialise for this call.
        chunks: list[str] = []
        usage = None
        response = None

        # Start timer for call
        started = time.perf_counter()

        # Attempt call.
        try:

            # Set kwargs for this call
            call_kwargs = dict(kwargs)

            # Add timeout kwarg if necessary
            if use_timeout:
                call_kwargs["timeout"] = timeout_s

            response = cast(
                Any, 
                litellm.completion(
                    model=model,
                    messages=messages,
                    stream_options={"include_usage": True},
                    extra_body={
                        "keep_alive": "24h",
                        "options": {"num_ctx": context_budget},
                    },
                    reasoning_effort = "medium" if bool_thinking else "none", # OpenAI / Gemini / DeepSeek API
                    output_config = { # Anthropic API
                        "effort": "medium" if bool_thinking else "low",
                    },
                    think = True if bool_thinking else False, # Ollama
                    **call_kwargs
                )
            )

            # Stream the response and collect chunks.
            for chunk in response:

                # Enforce the limit on the whole call, not just per read.
                if use_timeout and time.perf_counter() - started > timeout_s:
                    raise litellm.Timeout(
                        message=f"Exceeded {timeout_s}s", 
                        model=model,
                        llm_provider=llm_provider,
                    )

                # Capture usage if available
                if getattr(chunk, "usage", None):
                    usage = chunk.usage

                # Get the content delta from the chunk.
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    chunks.append(delta)

            break # We successfully streamed the response, so exit the retry loop.
        except litellm.Timeout as e:

            # Cancel the current attempt and retry if applicable.
            close = getattr(response, "close", None)
            if callable(close):
                close()

            # If we are not using a timeout, error immediately.
            if not use_timeout:
                attempt = max_attempts = 1 # this will trigger the error handling immediately

            # Warn the user.
            if attempt == max_attempts:
                print(
                    f"[red]Error: all {max_attempts} attempts timed out after {timeout_s}s each[/red]",
                    file=sys.stderr,
                )
                sys.exit(1)
            else:
                print(f"[yellow]Timeout occurred on attempt {attempt}/{max_attempts}: {e}[yellow]")

    # Combine the collected chunks into the final response text.
    # Strip code fences just in case.
    final_text = strip_fences("".join(chunks))

    # Print the elapsed time for the LLM call
    elapsed = time.perf_counter() - started

    # # Create full response object for metadata reporting.
    # response_from_chunks = litellm.stream_chunk_builder(chunks, messages=messages)

    # Create metadata dictionary for the LLM call.
    meta = {
        "elapsed": f"{elapsed:.2f}",
        "model_loaded": model_loaded,
        # "prompt_tokens": response_from_chunks.usage.prompt_tokens,
        # "completion_tokens": response_from_chunks.usage.completion_tokens
    }

    if usage:
        meta["prompt_tokens"] = usage.prompt_tokens
        meta["completion_tokens"] = usage.completion_tokens
        meta["total_tokens"] = usage.total_tokens

        print(f"[green]Actual prompt tokens: {usage.prompt_tokens:,}, Completion tokens: {usage.completion_tokens:,}, Total tokens: {usage.total_tokens:,}[/green]")

    return final_text, meta