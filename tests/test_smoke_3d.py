import json
import os
from smoothride.demo import scene as S


def test_smoke_3d_writes_valid_scene(tmp_path):
    from scripts.smoke_3d import run
    out = tmp_path / "scene.json"
    run(str(out), agents=4, peds=2, steps=10)
    assert out.exists()
    data = json.loads(out.read_text())
    S.validate_scene(data)                      # conforms to schema v1
    assert data["worlds"]["trained"]["summary"]["cars"] == 4
    car = data["worlds"]["trained"]["cars"][0]
    assert len(car["z"]) == len(car["lng"])     # z baked for every frame
