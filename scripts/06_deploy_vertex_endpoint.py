"""Deploy the fine-tune (LoRA adapter from 04_train_lora.py, applied at
inference time over its base model -- see README's "Compute" section for why
this isn't a merged checkpoint) to a Vertex AI Endpoint, so the fine-tuned
system is served on managed GCP infrastructure while everything upstream
(data prep, SFT synthesis, RAG index, LoRA training) runs on the RunPod pod.
The base model keeps being served locally via serve/vllm_base.sh -- only the
fine-tune goes to Vertex, since that's the half of the comparison that's
actually worth paying for managed-endpoint overhead on.

What this does:
1. Syncs models/lora_adapter/ (small, ~150MB) to GCS. The base model itself
   is downloaded fresh from HF at container start (see
   serve/vertex_entrypoint.sh) rather than round-tripped through GCS.
2. Builds serve/Dockerfile.vertex via Cloud Build and pushes it to Artifact
   Registry (the container is the same vLLM --enable-lora server as
   serve/vllm_finetuned.sh, wrapped to satisfy Vertex's custom-container
   contract -- see serve/vertex_entrypoint.sh).
3. Uploads a Vertex AI Model pointing at that image + GCS artifact.
4. Deploys it to a new Endpoint on a GPU machine.
5. Writes deploy/vertex_endpoint.json so eval/run_eval.py and
   07_teardown_vertex_endpoint.py can find the endpoint without re-pasting IDs.

Requires: `gcloud auth application-default login` done once, the GCP
project/bucket/Artifact Registry repo already created (see README), and
HF_TOKEN set in the environment (the container needs it to download the base
model at a speed that clears Vertex's startup health check -- run this via
`set -a; source .env; set +a; python scripts/06_deploy_vertex_endpoint.py`).

Usage:
    python scripts/06_deploy_vertex_endpoint.py \
        --project nfl-finetune-eval --bucket nfl-finetune-eval-models
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

from google.cloud import aiplatform

ROOT = Path(__file__).resolve().parent.parent
LORA_DIR = ROOT / "models" / "lora_adapter"
DEPLOY_DIR = ROOT / "deploy"
ENDPOINT_INFO_PATH = DEPLOY_DIR / "vertex_endpoint.json"


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def sync_model_to_gcs(bucket: str, region: str) -> str:
    if not LORA_DIR.exists():
        raise SystemExit(f"{LORA_DIR} not found -- run 04_train_lora.py first.")
    gcs_path = f"gs://{bucket}/lora_adapter"
    run(["gcloud", "storage", "rsync", "-r", str(LORA_DIR), gcs_path])
    return gcs_path


def build_and_push_image(project: str, region: str, repo: str, tag: str) -> str:
    image_uri = f"{region}-docker.pkg.dev/{project}/{repo}/finetuned-vllm:{tag}"
    run([
        "gcloud", "builds", "submit", str(ROOT / "serve"),
        f"--config={ROOT / 'serve' / 'cloudbuild.yaml'}",
        f"--substitutions=_IMAGE={image_uri}",
        f"--project={project}",
    ])
    return image_uri


def deploy(project: str, region: str, gcs_model_path: str, image_uri: str,
           machine_type: str, accelerator_type: str, accelerator_count: int,
           base_model: str) -> aiplatform.Endpoint:
    aiplatform.init(project=project, location=region)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit(
            "HF_TOKEN not set -- source .env first "
            "(set -a; source .env; set +a) so the container can authenticate "
            "its base-model download."
        )

    model = aiplatform.Model.upload(
        display_name="nfl-rulebook-finetuned",
        artifact_uri=gcs_model_path,
        serving_container_image_uri=image_uri,
        serving_container_ports=[8080],
        serving_container_predict_route="/v1/chat/completions",
        serving_container_health_route="/health",
        serving_container_environment_variables={
            "MAX_MODEL_LEN": "4096",
            "BASE_MODEL_DIR": base_model,
            "HF_TOKEN": hf_token,
        },
    )
    print(f"Uploaded model: {model.resource_name}")

    endpoint = model.deploy(
        deployed_model_display_name="nfl-rulebook-finetuned-v1",
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        min_replica_count=1,
        max_replica_count=1,
        sync=True,
    )
    print(f"Deployed endpoint: {endpoint.resource_name}")
    return model, endpoint


def main(project: str, region: str, bucket: str, repo: str, tag: str,
         machine_type: str, accelerator_type: str, accelerator_count: int,
         base_model: str) -> None:
    gcs_model_path = sync_model_to_gcs(bucket, region)
    image_uri = build_and_push_image(project, region, repo, tag)
    model, endpoint = deploy(
        project, region, gcs_model_path, image_uri,
        machine_type, accelerator_type, accelerator_count, base_model,
    )

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    ENDPOINT_INFO_PATH.write_text(json.dumps({
        "project": project,
        "region": region,
        "endpoint_resource_name": endpoint.resource_name,
        "model_resource_name": model.resource_name,
    }, indent=2))
    print(f"\nWrote {ENDPOINT_INFO_PATH}")
    print(
        "\nRemember this is billed per hour while deployed -- run "
        "scripts/07_teardown_vertex_endpoint.py when you're done evaluating."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="nfl-finetune-eval")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--bucket", default="nfl-finetune-eval-models")
    parser.add_argument("--repo", default="nfl-finetune-eval")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--machine-type", default="g2-standard-8")
    parser.add_argument("--accelerator-type", default="NVIDIA_L4")
    parser.add_argument("--accelerator-count", type=int, default=1)
    # Must match the exact base the adapter was trained on. Tried swapping to
    # the full-precision Qwen/Qwen2.5-7B-Instruct release to dodge a vLLM
    # version's dropped bitsandbytes support (see Dockerfile.vertex) -- that
    # deployed and responded, but produced degenerate output (endless single-
    # token repetition) specifically when the adapter was applied, while the
    # raw base model on the same endpoint answered normally. So Unsloth's
    # bnb-4bit repack isn't a drop-in stand-in for the vanilla release (likely
    # tokenizer/special-token or tensor-layout differences) -- fixed the vLLM
    # version instead (pinned in Dockerfile.vertex) rather than the base model.
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    args = parser.parse_args()
    main(args.project, args.region, args.bucket, args.repo, args.tag,
         args.machine_type, args.accelerator_type, args.accelerator_count,
         args.base_model)
