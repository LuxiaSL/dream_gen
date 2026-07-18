"""
Resume state - the dream picks up where it left off (SPEC-resume.md)

Local-first checkpoint under output/resume/:
    resume_state.json    - counters, prompt state, config fingerprint
    last_keyframe.png    - full-res image the next session seeds from

Contracts:
- save is atomic (tmp + rename) and never raises to callers
- load returns None on ANY problem (missing, corrupt, fingerprint mismatch,
  missing image) - a bad checkpoint degrades to a fresh start, never a crash
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

RESUME_STATE_VERSION = 1


class PromptResumeState(BaseModel):
    """Enough of CombinatorialPromptSystem to resume the same sentence."""

    template_id: str = ""
    components: dict[str, str] = Field(default_factory=dict)
    total_frames: int = 0
    total_mutations: int = 0
    frames_since_mutation: int = 0


class ResumeState(BaseModel):
    version: int = RESUME_STATE_VERSION
    saved_at: float = 0.0
    session_id: str = ""  # chronicle session that wrote this checkpoint
    # Config fingerprint - mismatch means the saved image/state don't apply
    resolution: list[int] = Field(default_factory=list)
    model: str = ""
    # Continuity
    lifetime_keyframes: int = 0  # epoch + local keyframe at save time
    local_keyframe: int = 0
    prompt: PromptResumeState = Field(default_factory=PromptResumeState)
    last_keyframe_file: str = "last_keyframe.png"  # relative to resume dir


def _fingerprint(config: dict) -> tuple[list[int], str]:
    gen = config.get("generation", {})
    resolution = list(gen.get("resolution", []))
    model = str(gen.get("model", ""))
    return resolution, model


def save_resume_state(
    resume_dir: Path,
    *,
    config: dict,
    session_id: str,
    lifetime_keyframes: int,
    local_keyframe: int,
    prompt_state: PromptResumeState,
    keyframe_image_path: Path,
) -> bool:
    """
    Write a checkpoint atomically. Returns True on success, never raises.

    Intended to run in an executor - it does blocking file I/O.
    """
    try:
        resume_dir.mkdir(parents=True, exist_ok=True)
        resolution, model = _fingerprint(config)

        # Copy the keyframe first (state without image is useless; the
        # reverse - image without state - is merely ignored at load)
        img_tmp = resume_dir / "last_keyframe.png.tmp"
        img_final = resume_dir / "last_keyframe.png"
        shutil.copyfile(keyframe_image_path, img_tmp)
        img_tmp.replace(img_final)

        state = ResumeState(
            saved_at=time.time(),
            session_id=session_id,
            resolution=resolution,
            model=model,
            lifetime_keyframes=lifetime_keyframes,
            local_keyframe=local_keyframe,
            prompt=prompt_state,
            last_keyframe_file=img_final.name,
        )
        state_tmp = resume_dir / "resume_state.json.tmp"
        state_tmp.write_text(state.model_dump_json(indent=2))
        state_tmp.replace(resume_dir / "resume_state.json")
        return True
    except Exception:
        logger.debug("Resume checkpoint save failed", exc_info=True)
        return False


def load_resume_state(resume_dir: Path, config: dict) -> Optional[ResumeState]:
    """
    Load and validate a checkpoint. Returns None on any problem.
    """
    state_path = resume_dir / "resume_state.json"
    try:
        if not state_path.is_file():
            return None

        state = ResumeState(**json.loads(state_path.read_text()))

        if state.version != RESUME_STATE_VERSION:
            logger.info(
                f"Resume: checkpoint version {state.version} != "
                f"{RESUME_STATE_VERSION}, starting fresh"
            )
            return None

        resolution, model = _fingerprint(config)
        if state.resolution != resolution or state.model != model:
            logger.info(
                f"Resume: config fingerprint changed "
                f"({state.resolution}/{state.model} -> {resolution}/{model}), "
                f"starting fresh"
            )
            return None

        image_path = resume_dir / state.last_keyframe_file
        if not image_path.is_file() or image_path.stat().st_size == 0:
            logger.info("Resume: checkpoint image missing, starting fresh")
            return None

        # Light image sanity check - a truncated PNG would poison bootstrap
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            logger.info("Resume: checkpoint image corrupt, starting fresh")
            return None

        age_h = (time.time() - state.saved_at) / 3600
        logger.info(
            f"Resume: valid checkpoint found (saved {age_h:.1f}h ago, "
            f"lifetime kf {state.lifetime_keyframes}, "
            f"template '{state.prompt.template_id}')"
        )
        return state
    except Exception:
        logger.warning(f"Resume: failed to load {state_path}, starting fresh")
        return None


def resume_image_path(resume_dir: Path, state: ResumeState) -> Path:
    return resume_dir / state.last_keyframe_file
