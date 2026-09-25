"""
Tests for mutation reach + anti-repeat (prompts/combinatorial.py, 2026-09-25).

Chronicle data showed mutations could only ever touch color/atmosphere/light/
temporal/texture (subject, medium, composition frozen per era) and ~20% of
mutations returned to a recently held word. These run the real prompt system
over the real vocabulary and the shipping b200 config.
"""

from collections import Counter
from pathlib import Path

import pytest
import yaml

from prompts.combinatorial import CombinatorialPromptSystem

CONFIG = Path(__file__).resolve().parents[1] / "config.b200.yaml"
STRUCTURAL = {"subject_form", "material_substance", "medium_render",
              "spatial_logic", "setting_location", "phenomenon_pattern",
              "scale_perspective"}


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(CONFIG))


def make(cfg, template):
    ps = CombinatorialPromptSystem(config=cfg)
    ps.switch_template(template)
    return ps


def categories_mutated(ps, n):
    seen = Counter()
    for _ in range(n):
        before = {k: v.word for k, v in ps.current_components.items()}
        ps.mutate()
        after = {k: v.word for k, v in ps.current_components.items()}
        for k in after:
            if before.get(k) != after[k]:
                seen[k] += 1
    return seen


def test_shipping_config_lets_mutations_reach_structure(cfg):
    weights = cfg["fresh_generation"]["mutation"]["category_weights"]
    assert STRUCTURAL & set(weights), "no structural category is mutable"


def test_specimen_era_changes_more_than_its_color(cfg):
    # specimen used to have exactly one mutable slot: color_logic
    seen = categories_mutated(make(cfg, "specimen"), 200)
    assert set(seen) & STRUCTURAL
    assert seen["color_logic"] < 200 * 0.8


def test_structural_share_stays_a_minority(cfg):
    # reach, not churn: the era should keep a recognizable identity
    for template in ("process_state", "environmental", "liminal"):
        seen = categories_mutated(make(cfg, template), 400)
        structural = sum(v for k, v in seen.items() if k in STRUCTURAL)
        assert 0.10 < structural / sum(seen.values()) < 0.55, (template, seen)


def test_no_return_to_recent_words(cfg):
    ps = make(cfg, "process_state")
    n = ps.recent_word_memory
    assert n >= 4
    history = {}
    for _ in range(300):
        before = {k: v.word for k, v in ps.current_components.items()}
        ps.mutate()
        for k, v in ps.current_components.items():
            if before.get(k) != v.word:
                past = history.setdefault(k, [before[k]])
                assert v.word not in past[-n:], f"{k} returned to '{v.word}'"
                past.append(v.word)


def test_template_switch_clears_repeat_history(cfg):
    ps = make(cfg, "process_state")
    for _ in range(20):
        ps.mutate()
    assert ps._recent_words
    ps.switch_template("liminal")
    assert not ps._recent_words


def test_memory_off_restores_legacy_behavior(cfg):
    legacy = dict(cfg)
    legacy["fresh_generation"] = dict(cfg["fresh_generation"])
    legacy["fresh_generation"]["mutation"] = dict(
        cfg["fresh_generation"]["mutation"], recent_word_memory=0)
    ps = make(legacy, "process_state")
    for _ in range(30):
        ps.mutate()
    assert not ps._recent_words
