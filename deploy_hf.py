"""Deploy current directory as a new or existing Docker Space.
Usage: HF_TOKEN=hf_... python deploy_hf.py USERNAME/SPACE_NAME
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

if len(sys.argv) != 2 or "/" not in sys.argv[1]:
    raise SystemExit("Usage: HF_TOKEN=hf_... python deploy_hf.py USERNAME/SPACE_NAME")
token = os.getenv("HF_TOKEN")
if not token:
    raise SystemExit("Set HF_TOKEN (HF account write token) first")
api = HfApi(token=token)
repo_id = sys.argv[1]
api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(Path(__file__).parent), ignore_patterns=[".git/*", "*.pyc", "__pycache__/*", "tests/*", ".env", "*.zip"])
print(f"Uploaded: https://huggingface.co/spaces/{repo_id}")
print("Now set GATEWAY_API_KEY (and optional UPSTREAM_HF_TOKEN) in Space Settings > Secrets")
