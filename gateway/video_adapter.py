"""Route video models without changing the existing Wan contract."""
from . import wan2_2_i2v_v3 as wan
from . import minimax_h3_workflow as h3

MODELS = (wan.MODEL_ID, h3.MODEL_ID)

def validate(payload, paths):
    if payload["model"] == h3.MODEL_ID:
        return h3.build_args(payload, paths)
    return wan.build_kwargs(payload, paths)

def run(payload, token, paths, ip_token=None):
    adapter = h3 if payload["model"] == h3.MODEL_ID else wan
    return adapter.run(payload, token, paths, ip_token=ip_token)
