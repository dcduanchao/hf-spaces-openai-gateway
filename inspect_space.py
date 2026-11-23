"""Usage: python inspect_space.py owner/space; authenticates with UPSTREAM_HF_TOKEN when set."""
import os
import sys
from gradio_client import Client

if len(sys.argv) != 2 or "/" not in sys.argv[1]:
    raise SystemExit("Usage: python inspect_space.py owner/space")
client = Client(sys.argv[1], token=os.getenv("UPSTREAM_HF_TOKEN") or False)
try:
    client.view_api()
finally:
    client.close()
