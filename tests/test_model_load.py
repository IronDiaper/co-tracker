import hashlib
import json
import os
import socket
import time
import warnings

import torch

from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from helpers import RESULTS_DIR, run_demo

CKPT_SIZE = 101890938
CKPT_SHA256 = "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"


def test_checkpoint_is_local(report):
    path = run_demo.DEFAULT_CHECKPOINT
    assert os.path.isfile(path), f"нет чекпоинта {path}"
    assert os.path.getsize(path) == CKPT_SIZE
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    assert h.hexdigest() == CKPT_SHA256
    report(f"{path}: {CKPT_SIZE} bytes, sha256 {CKPT_SHA256[:12]}... OK")


def test_model_loads_without_network(monkeypatch, report):
    def forbidden(*args, **kwargs):
        raise AssertionError("попытка обращения к сети или torch.hub при загрузке модели")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(torch.hub, "load", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)

    t0 = time.time()
    predictor = run_demo.load_model(run_demo.DEFAULT_CHECKPOINT, "cpu")
    load_seconds = time.time() - t0

    assert isinstance(predictor.model, CoTrackerThreeOffline)
    assert tuple(predictor.interp_shape) == (384, 512)
    assert not predictor.model.training

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = torch.load(run_demo.DEFAULT_CHECKPOINT, map_location="cpu")
    state = state.get("model", state)
    model_state = predictor.model.state_dict()
    assert set(state) == set(model_state)
    assert all(torch.equal(model_state[k], state[k]) for k in state)

    n_params = sum(p.numel() for p in predictor.model.parameters())
    info = {"class": type(predictor.model).__name__, "parameters": n_params,
            "model_resolution": list(predictor.interp_shape),
            "window_len": predictor.model.window_len, "load_seconds": round(load_seconds, 2)}
    path = os.path.join(RESULTS_DIR, "model_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    report(f"{info['class']}, {n_params / 1e6:.1f}M params, {len(state)} tensors match the file, "
           f"loaded in {load_seconds:.1f} s, network blocked",
           artifact=path, parameters=n_params, load_seconds=load_seconds)
