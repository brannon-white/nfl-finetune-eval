"""Undeploy and delete the Vertex AI endpoint/model from
06_deploy_vertex_endpoint.py. GPU-backed endpoints bill per hour whether or
not you're actively querying them -- run this as soon as eval/run_eval.py is
done with the finetuned_rag / finetuned systems.

Usage:
    python scripts/07_teardown_vertex_endpoint.py
"""
import json
from pathlib import Path

from google.cloud import aiplatform

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT_INFO_PATH = ROOT / "deploy" / "vertex_endpoint.json"


def main() -> None:
    if not ENDPOINT_INFO_PATH.exists():
        raise SystemExit(f"{ENDPOINT_INFO_PATH} not found -- nothing to tear down.")

    info = json.loads(ENDPOINT_INFO_PATH.read_text())
    aiplatform.init(project=info["project"], location=info["region"])

    endpoint = aiplatform.Endpoint(info["endpoint_resource_name"])
    print(f"Undeploying all models from {endpoint.resource_name}...")
    endpoint.undeploy_all(sync=True)
    endpoint.delete()
    print("Endpoint deleted.")

    model = aiplatform.Model(info["model_resource_name"])
    model.delete()
    print("Model resource deleted.")

    ENDPOINT_INFO_PATH.unlink()
    print(f"Removed {ENDPOINT_INFO_PATH}. GPU billing for this endpoint has stopped.")


if __name__ == "__main__":
    main()
