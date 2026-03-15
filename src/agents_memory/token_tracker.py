"""Monkey-patch OpenAI client to count LLM calls and tokens."""

import threading

import openai.resources.chat.completions

_lock = threading.Lock()
_stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
_original_create = None


class _StreamWrapper:
    """Wrap a streaming response to capture usage from the final chunk."""

    def __init__(self, stream):
        self._stream = stream

    def __iter__(self):
        for chunk in self._stream:
            if chunk.usage:
                with _lock:
                    _stats["prompt_tokens"] += chunk.usage.prompt_tokens or 0
                    _stats["completion_tokens"] += chunk.usage.completion_tokens or 0
            yield chunk

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *args):
        return self._stream.__exit__(*args)


def _tracked_create(*args, **kwargs):
    is_stream = kwargs.get("stream", False)

    # For streaming: request usage in final chunk
    if is_stream:
        kwargs.setdefault("stream_options", {})["include_usage"] = True

    response = _original_create(*args, **kwargs)

    with _lock:
        _stats["calls"] += 1

    if is_stream:
        return _StreamWrapper(response)

    if hasattr(response, "usage") and response.usage:
        with _lock:
            _stats["prompt_tokens"] += response.usage.prompt_tokens or 0
            _stats["completion_tokens"] += response.usage.completion_tokens or 0

    return response


def start():
    """Patch OpenAI client to start tracking."""
    global _original_create
    if _original_create is None:
        _original_create = openai.resources.chat.completions.Completions.create
        openai.resources.chat.completions.Completions.create = _tracked_create


def get_stats():
    """Return current stats dict."""
    with _lock:
        return {
            "calls": _stats["calls"],
            "prompt_tokens": _stats["prompt_tokens"],
            "completion_tokens": _stats["completion_tokens"],
            "total_tokens": _stats["prompt_tokens"] + _stats["completion_tokens"],
        }


def reset():
    """Reset counters to zero."""
    with _lock:
        _stats["calls"] = 0
        _stats["prompt_tokens"] = 0
        _stats["completion_tokens"] = 0
