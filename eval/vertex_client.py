"""Query the Vertex AI-hosted fine-tuned endpoint (see
scripts/06_deploy_vertex_endpoint.py) with the same OpenAI chat-completions
shape used against the RunPod-hosted vLLM servers, via Endpoint.raw_predict --
the serving container is vLLM's own OpenAI-compatible server, so no request/
response translation is needed, just an auth'd HTTP call instead of a plain
one.
"""
import json
from pathlib import Path

from google.cloud import aiplatform

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT_INFO_PATH = ROOT / "deploy" / "vertex_endpoint.json"


def load_endpoint() -> aiplatform.Endpoint:
    if not ENDPOINT_INFO_PATH.exists():
        raise SystemExit(
            f"{ENDPOINT_INFO_PATH} not found -- run "
            "scripts/06_deploy_vertex_endpoint.py first."
        )
    info = json.loads(ENDPOINT_INFO_PATH.read_text())
    aiplatform.init(project=info["project"], location=info["region"])
    return aiplatform.Endpoint(info["endpoint_resource_name"])


def query_vertex(endpoint: aiplatform.Endpoint, model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 300,
    }).encode("utf-8")

    response = endpoint.raw_predict(body=body, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
