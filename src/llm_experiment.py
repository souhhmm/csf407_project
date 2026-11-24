import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
KEY_FILE = Path.home() / ".config" / "csf407" / "openrouter_key"
# Tiers chosen with the user: very cheap, cheap open-weight, medium.
MODELS = {
    "very_cheap": "google/gemini-3.1-flash-lite",
    "cheap_open_weight": "z-ai/glm-5.2",
    "medium": "anthropic/claude-sonnet-5.5",
}
BUDGET_USD = 4.0
MAX_TOKENS = 4000


def read_key() -> str:
    """Read the OpenRouter key from the private key file, or from the environment.

    Returns:
        The key. It is never printed or logged.

    Raises:
        RuntimeError: If no key is found.
    """
    import os

    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"].strip()
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip()
    raise RuntimeError("No OpenRouter key at " + str(KEY_FILE))


def ask(model: str, system: str, prompt: str, temperature: float, key: str) -> dict:
    """Send one chat request and return the reply with token usage and cost.

    Args:
        model: OpenRouter model id.
        system: System message.
        prompt: User message.
        temperature: Sampling temperature.
        key: OpenRouter key.

    Returns:
        A dict with text, usage, cost (USD) and seconds.

    Raises:
        RuntimeError: If the request keeps failing.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": MAX_TOKENS,
            "usage": {"include": True},
        }
    ).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    last = ""
    for attempt in range(3):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.loads(response.read())
            usage = data.get("usage", {})
            return {
                "text": data["choices"][0]["message"].get("content") or "",
                "usage": usage,
                "cost": float(usage.get("cost", 0.0)),
                "seconds": time.perf_counter() - started,
            }
        except urllib.error.HTTPError as error:
            last = "HTTP " + str(error.code) + ": " + error.read().decode()[:200]
            if error.code in (401, 402, 403):
                break
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as error:
            last = str(error)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("OpenRouter request failed: " + last)


def run_experiment(
    prompts: List[Dict[str, str]],
    output_path: Path,
    models: Optional[Dict[str, str]] = None,
    samples: int = 3,
    temperature: float = 0.7,
    system: str = "You are a careful data analyst.",
    budget: float = BUDGET_USD,
) -> float:
    """Ask every model every prompt several times, appending replies to a JSONL file.

    Replies already in the file (same model, prompt id and sample) are skipped,
    so an interrupted run resumes without paying twice.

    Args:
        prompts: Dicts with id and text.
        output_path: JSONL file of replies.
        models: Tier name to model id; defaults to MODELS.
        samples: Replies per model and prompt.
        temperature: Sampling temperature.
        system: System message.
        budget: Stop asking once the total spend reaches this many USD.

    Returns:
        Total spend in USD recorded in the file.
    """
    models = models or MODELS
    key = read_key()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    spent = 0.0
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            row = json.loads(line)
            done.add((row["model"], row["prompt_id"], row["sample"]))
            spent += row["cost"]
    for prompt in prompts:
        for tier, model in models.items():
            for sample in range(samples):
                if (model, prompt["id"], sample) in done:
                    continue
                if spent >= budget:
                    print("Budget reached; stopping.", flush=True)
                    return spent
                reply = ask(model, system, prompt["text"], temperature, key)
                spent += reply["cost"]
                row = {
                    "tier": tier,
                    "model": model,
                    "prompt_id": prompt["id"],
                    "sample": sample,
                    **reply,
                }
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(
                    prompt["id"]
                    + " | "
                    + model
                    + " | "
                    + str(sample)
                    + " | $"
                    + format(reply["cost"], ".4f")
                    + " | total $"
                    + format(spent, ".3f"),
                    flush=True,
                )
    return spent
