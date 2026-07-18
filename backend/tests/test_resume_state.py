"""
Tests for resume state (SPEC-resume.md)

Contracts:
- save is atomic, load round-trips
- load returns None (never raises) on: missing file, corrupt json,
  version mismatch, config fingerprint mismatch, missing/corrupt image
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from core.resume_state import (
    PromptResumeState,
    ResumeState,
    load_resume_state,
    resume_image_path,
    save_resume_state,
)


def make_config(resolution=None, model="sd15") -> dict:
    return {"generation": {"resolution": resolution or [1024, 512], "model": model}}


@pytest.fixture
def keyframe_png(tmp_path: Path) -> Path:
    path = tmp_path / "keyframe_0042.png"
    Image.new("RGB", (64, 32), color=(120, 40, 200)).save(path)
    return path


def make_prompt_state() -> PromptResumeState:
    return PromptResumeState(
        template_id="material_study",
        components={"color_logic": "verdigris", "material_substance": "bismuth"},
        total_frames=812,
        total_mutations=31,
        frames_since_mutation=7,
    )


def test_save_load_roundtrip(tmp_path, keyframe_png):
    resume_dir = tmp_path / "resume"
    config = make_config()

    ok = save_resume_state(
        resume_dir,
        config=config,
        session_id="abc123",
        lifetime_keyframes=15000,
        local_keyframe=42,
        prompt_state=make_prompt_state(),
        keyframe_image_path=keyframe_png,
    )
    assert ok

    state = load_resume_state(resume_dir, config)
    assert state is not None
    assert state.lifetime_keyframes == 15000
    assert state.local_keyframe == 42
    assert state.session_id == "abc123"
    assert state.prompt.template_id == "material_study"
    assert state.prompt.components["color_logic"] == "verdigris"
    assert state.prompt.frames_since_mutation == 7

    img = resume_image_path(resume_dir, state)
    assert img.is_file()
    assert Image.open(img).size == (64, 32)

    # No tmp files left behind
    assert not list(resume_dir.glob("*.tmp"))


def test_load_missing_returns_none(tmp_path):
    assert load_resume_state(tmp_path / "nope", make_config()) is None


def test_load_corrupt_json_returns_none(tmp_path):
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    (resume_dir / "resume_state.json").write_text("{ not json !!!")
    assert load_resume_state(resume_dir, make_config()) is None


def test_fingerprint_mismatch_returns_none(tmp_path, keyframe_png):
    resume_dir = tmp_path / "resume"
    save_resume_state(
        resume_dir, config=make_config(resolution=[1024, 512]),
        session_id="s", lifetime_keyframes=1, local_keyframe=1,
        prompt_state=make_prompt_state(), keyframe_image_path=keyframe_png,
    )
    # Resolution changed -> saved image no longer applies
    assert load_resume_state(resume_dir, make_config(resolution=[512, 512])) is None
    # Model changed -> same
    assert load_resume_state(resume_dir, make_config(model="flux")) is None
    # Unchanged -> loads
    assert load_resume_state(resume_dir, make_config()) is not None


def test_version_mismatch_returns_none(tmp_path, keyframe_png):
    resume_dir = tmp_path / "resume"
    config = make_config()
    save_resume_state(
        resume_dir, config=config, session_id="s",
        lifetime_keyframes=1, local_keyframe=1,
        prompt_state=make_prompt_state(), keyframe_image_path=keyframe_png,
    )
    data = json.loads((resume_dir / "resume_state.json").read_text())
    data["version"] = 999
    (resume_dir / "resume_state.json").write_text(json.dumps(data))
    assert load_resume_state(resume_dir, config) is None


def test_missing_image_returns_none(tmp_path, keyframe_png):
    resume_dir = tmp_path / "resume"
    config = make_config()
    save_resume_state(
        resume_dir, config=config, session_id="s",
        lifetime_keyframes=1, local_keyframe=1,
        prompt_state=make_prompt_state(), keyframe_image_path=keyframe_png,
    )
    (resume_dir / "last_keyframe.png").unlink()
    assert load_resume_state(resume_dir, config) is None


def test_corrupt_image_returns_none(tmp_path, keyframe_png):
    resume_dir = tmp_path / "resume"
    config = make_config()
    save_resume_state(
        resume_dir, config=config, session_id="s",
        lifetime_keyframes=1, local_keyframe=1,
        prompt_state=make_prompt_state(), keyframe_image_path=keyframe_png,
    )
    (resume_dir / "last_keyframe.png").write_bytes(b"not a png")
    assert load_resume_state(resume_dir, config) is None


def test_save_missing_source_image_fails_cleanly(tmp_path):
    ok = save_resume_state(
        tmp_path / "resume", config=make_config(), session_id="s",
        lifetime_keyframes=1, local_keyframe=1,
        prompt_state=make_prompt_state(),
        keyframe_image_path=tmp_path / "gone.png",
    )
    assert ok is False
    # A failed save must not leave a loadable checkpoint
    assert load_resume_state(tmp_path / "resume", make_config()) is None


def test_checkpoint_overwrites_previous(tmp_path, keyframe_png):
    resume_dir = tmp_path / "resume"
    config = make_config()
    for kf in (50, 100):
        save_resume_state(
            resume_dir, config=config, session_id="s",
            lifetime_keyframes=kf, local_keyframe=kf,
            prompt_state=make_prompt_state(), keyframe_image_path=keyframe_png,
        )
    state = load_resume_state(resume_dir, config)
    assert state is not None and state.lifetime_keyframes == 100
