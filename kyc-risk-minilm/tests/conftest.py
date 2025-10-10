import os
import pytest


@pytest.fixture(autouse=True, scope="session")
def set_env_model_ckpt():
    # Point API to a local checkpoint inside repo for tests
    # Prefer api/minilm_cls_best.pt if present, else root.
    candidates = [
        os.path.join("api", "minilm_cls_best.pt"),
        os.path.join("minilm_cls_best.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            os.environ["MODEL_CKPT"] = os.path.abspath(p)
            break
    # set audit secret for deterministic HMAC
    os.environ.setdefault("AUDIT_HMAC", "test_secret")
    yield

