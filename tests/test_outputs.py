import os

import numpy as np
import pytest
import torch

import synthetic
from helpers import RESULTS_DIR, plot_points_strip, predict

T, H, W, GRID = 24, 256, 256, 8
TOLERANCE_PX = 2.0


@pytest.fixture(scope="module")
def grid_run(model, device):
    video = synthetic.make_shift_video(T, H, W, dx=3, dy=2, seed=7)
    tracks, visibility = predict(model, video, grid_size=GRID, device=device)
    return video, tracks, visibility


def test_output_shapes_and_types(grid_run, report):
    _, tracks, visibility = grid_run
    B, N = 1, GRID * GRID
    assert tracks.shape == (B, T, N, 2)
    assert visibility.shape == (B, T, N)
    assert tracks.dtype == torch.float32
    assert visibility.dtype == torch.bool
    report(f"tracks {tuple(tracks.shape)} {tracks.dtype}, visibility {tuple(visibility.shape)} "
           f"{visibility.dtype}")


def test_output_values_finite_and_in_frame(grid_run, report):
    video, tracks, visibility = grid_run
    tracks, visibility = tracks[0].numpy(), visibility[0].numpy()
    assert np.isfinite(tracks).all()

    x, y = tracks[..., 0], tracks[..., 1]
    outside = ((x < -TOLERANCE_PX) | (x > W - 1 + TOLERANCE_PX)
               | (y < -TOLERANCE_PX) | (y > H - 1 + TOLERANCE_PX))
    assert not (outside & visibility).any(), "видимая точка вне кадра"
    assert visibility[0].all(), "на кадре запроса все точки видимы"

    path = os.path.join(RESULTS_DIR, "outputs_grid.png")
    plot_points_strip(path, video, tracks, visibility, picks=[0, T // 2, T - 1],
                      title="Сетка 8×8 на сдвигающейся текстуре: зелёные — видимы, "
                            "красные контуры — вышли из кадра")
    report(f"all values finite; visible points inside frame (+/-{TOLERANCE_PX:g} px); "
           f"{outside.sum()} predictions outside frame, all invisible",
           artifact=path, outside_invisible=int(outside.sum()))
