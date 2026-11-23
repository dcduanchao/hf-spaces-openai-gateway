"""Attach the private history bucket while preserving existing Space volumes.

Uses the deployment HF_TOKEN; requires permission to edit Space settings and
read the bucket. Does not expose the token in logs.
"""
import os
from huggingface_hub import HfApi, Volume

SPACE = os.environ.get("HF_SPACE_ID", "duan0320/hf-spaces-openai-gateway")
SOURCE = "duan0320/image-history"
DEST = "/data"


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required in the GitHub Actions secret")
    api = HfApi(token=token)
    current = list(api.get_space_runtime(repo_id=SPACE).volumes or [])
    for volume in current:
        if volume.mount_path == DEST:
            if volume.type == "bucket" and volume.source == SOURCE and not volume.read_only:
                print(f"History bucket already mounted read-write at {DEST}")
                return
            raise RuntimeError(f"Mount path {DEST} is already occupied; refusing to replace it")
    updated = current + [Volume(type="bucket", source=SOURCE, mount_path=DEST, read_only=False)]
    api.set_space_volumes(repo_id=SPACE, volumes=updated)
    print(f"Mounted private history bucket {SOURCE} at {DEST}; preserved {len(current)} previous volume(s)")


if __name__ == "__main__":
    main()
