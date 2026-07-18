"""
Combinatorial Prompt System
Generates billions of unique prompts through template + component combination

Key Features:
- Templates with slot placeholders for components
- Multi-component selection: {category:N} picks N different items without replacement
- Semantic opposite generation for negative prompts
- Component mutations with configurable probability and category weighting
- Similarity-guided selection using pre-computed embeddings (Phase 6)

Syntax:
    {category}      → Pick 1 random component
    {category:N}    → Pick N different components, space-joined
    {category:N:,}  → Pick N different components, joined by ", "
"""

import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Component:
    """Single component with dual CLIP/T5 embeddings and optional semantic opposite"""
    word: str
    clip_embedding: List[float]
    t5_embedding: List[float]
    opposite: Optional[str] = None


@dataclass
class Template:
    """Template structure with slot definitions"""
    id: str
    structure: str
    required_components: List[str]
    notes: str = ""
    
    # Parsed slots: list of (category, count, separator)
    slots: List[Tuple[str, int, str]] = field(default_factory=list)
    
    def __post_init__(self):
        """Parse slots from structure string"""
        self.slots = self._parse_slots(self.structure)
    
    def _parse_slots(self, structure: str) -> List[Tuple[str, int, str]]:
        """
        Parse slot placeholders from template structure
        
        Syntax:
            {category}      → (category, 1, " ")
            {category:N}    → (category, N, " ")
            {category:N:,}  → (category, N, ", ")
        
        Returns:
            List of (category, count, separator) tuples
        """
        slots = []
        # Match {word} or {word:N} or {word:N:sep}
        pattern = r'\{(\w+)(?::(\d+))?(?::([^}]*))?\}'
        
        for match in re.finditer(pattern, structure):
            category = match.group(1)
            count = int(match.group(2)) if match.group(2) else 1
            separator = match.group(3) if match.group(3) is not None else " "
            slots.append((category, count, separator))
        
        return slots


class CombinatorialPromptSystem:
    """
    Manages combinatorial prompt generation with templates and component pools
    
    Responsibilities:
    - Load templates and components from YAML
    - Generate prompts by filling template slots with components
    - Generate negative prompts from semantic opposites
    - Handle component mutations (random chance per category)
    - Track current template and component state
    - Provide embeddings for similarity calculations
    
    Usage:
        prompt_system = CombinatorialPromptSystem(
            templates_path="prompts/templates.yaml",
            components_path="prompts/components.yaml"
        )
        
        # Generate prompts
        positive = prompt_system.get_next_prompt()
        negative = prompt_system.get_negative_prompt()
        
        # Check for mutations
        if prompt_system.should_mutate():
            prompt_system.mutate()
    """
    
    # Categories that get semantic opposites in negative prompts
    NEGATABLE_CATEGORIES = {
        'color_logic', 'light_behavior', 'atmosphere_field',
        'temporal_state', 'texture_density', 'medium_render'
    }
    
    def __init__(
        self,
        templates_path: Optional[str] = None,
        components_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize combinatorial prompt system
        
        Args:
            templates_path: Path to templates YAML file
            components_path: Path to components YAML file
            config: Optional config dict with mutation settings
        """
        # Resolve paths relative to project root
        project_root = Path(__file__).parent.parent.parent
        
        self.templates_path = Path(templates_path) if templates_path else project_root / "prompts" / "templates.yaml"
        self.components_path = Path(components_path) if components_path else project_root / "prompts" / "components.yaml"
        
        # Load data
        self.templates: Dict[str, Template] = {}
        self.components: Dict[str, List[Component]] = {}
        self._load_templates()
        self._load_components()
        
        # === Current State ===
        self.current_template: Optional[Template] = None
        self.current_components: Dict[str, Component] = {}  # {category: selected_component}
        self.current_prompt: Optional[str] = None
        self.current_negative: Optional[str] = None
        
        # Multi-component state (for {category:N} slots)
        # {category: [component1, component2, ...]} - tracks all selected for current prompt
        self.current_multi_components: Dict[str, List[Component]] = {}
        
        # === Mutation Configuration ===
        # Defaults (can be overridden by config)
        self.mutation_probability = 0.12
        self.staleness_threshold = 25
        self.category_weights = {
            'color_logic': 0.35,
            'atmosphere_field': 0.25,
            'light_behavior': 0.20,
            'temporal_state': 0.15,
            'texture_density': 0.05,
        }
        self.similarity_target = 0.55
        self.similarity_range = 0.25
        
        # Override with config if provided
        if config and 'fresh_generation' in config:
            mutation_config = config['fresh_generation'].get('mutation', {})
            self.mutation_probability = mutation_config.get('base_probability', self.mutation_probability)
            self.staleness_threshold = mutation_config.get('staleness_threshold', self.staleness_threshold)
            if 'category_weights' in mutation_config:
                self.category_weights = mutation_config['category_weights']
            self.similarity_target = mutation_config.get('similarity_target', self.similarity_target)
            self.similarity_range = mutation_config.get('similarity_range', self.similarity_range)
        
        # === State Tracking ===
        self.frames_since_mutation = 0
        self.total_frames = 0
        self.total_mutations = 0
        self.in_bend_mode = False
        self.bend_frames_remaining = 0
        
        # === DRIFT/BEND mode (for Phase 2 integration) ===
        self.bend_duration = 4  # Frames to stay in bend mode after mutation
        
        # Initialize with random template and components
        self._initialize_random_state()
        
        logger.info(f"CombinatorialPromptSystem initialized")
        logger.info(f"  Templates: {len(self.templates)}")
        logger.info(f"  Component categories: {list(self.components.keys())}")
        logger.info(f"  Mutation probability: {self.mutation_probability:.1%}")
    
    def _load_templates(self) -> None:
        """Load templates from YAML file"""
        if not self.templates_path.exists():
            raise FileNotFoundError(f"Templates file not found: {self.templates_path}")
        
        with open(self.templates_path, 'r') as f:
            data = yaml.safe_load(f)
        
        for template_data in data.get('templates', []):
            template = Template(
                id=template_data['id'],
                structure=template_data['structure'],
                required_components=template_data['required_components'],
                notes=template_data.get('notes', '')
            )
            self.templates[template.id] = template
        
        logger.debug(f"Loaded {len(self.templates)} templates")
    
    def _load_components(self) -> None:
        """
        Load components from YAML file and embeddings from companion npz file

        Architecture:
        - components.yaml: Human-editable file with word + opposite
        - components_embeddings.npz: Dual-space embeddings (CLIP 512-dim + T5 1024-dim)

        NPZ key format: "category/word/clip" and "category/word/t5"
        Falls back to legacy single-vector format ("category/word") for compatibility.

        If npz file doesn't exist or is outdated, embeddings will be empty
        and similarity-guided selection will fall back to random.
        """
        if not self.components_path.exists():
            raise FileNotFoundError(f"Components file not found: {self.components_path}")

        # Load base component data
        with open(self.components_path, 'r') as f:
            data = yaml.safe_load(f)

        # Try to load embeddings from companion npz file
        embeddings_path = self.components_path.parent / "components_embeddings.npz"
        clip_dict, t5_dict = self._load_embeddings_npz(embeddings_path)

        # Build component objects
        embedded_count = 0
        for category, items in data.get('components', {}).items():
            self.components[category] = []
            for item in items:
                key = f"{category}/{item['word']}"

                clip_emb = clip_dict.get(key, [])
                t5_emb = t5_dict.get(key, [])

                if clip_emb is not None and len(clip_emb) > 0:
                    embedded_count += 1

                self.components[category].append(
                    Component(
                        word=item['word'],
                        clip_embedding=clip_emb if isinstance(clip_emb, list) else clip_emb.tolist(),
                        t5_embedding=t5_emb if isinstance(t5_emb, list) else t5_emb.tolist(),
                        opposite=item.get('opposite')
                    )
                )

        # Log loading summary
        total_components = sum(len(v) for v in self.components.values())
        logger.debug(f"Loaded components: {', '.join(f'{k}={len(v)}' for k, v in self.components.items())}")
        if embedded_count > 0:
            logger.info(
                f"Loaded {embedded_count}/{total_components} dual embeddings "
                f"(CLIP {len(next((v for v in clip_dict.values()), []))}d, "
                f"T5 {len(next((v for v in t5_dict.values()), []))}d) "
                f"from {embeddings_path.name}"
            )
        else:
            logger.info("No embeddings loaded (using random selection for mutations)")

    def _load_embeddings_npz(self, path: Path) -> tuple:
        """
        Load dual-space embeddings from npz file

        Supports two key formats:
        - Dual-space: "category/word/clip" + "category/word/t5" (preferred)
        - Legacy single-vector: "category/word" (loaded as CLIP only)

        Returns:
            Tuple of (clip_dict, t5_dict), each mapping "category/word" to numpy array.
            Returns ({}, {}) if file not found or load fails.
        """
        import numpy as np

        if not path.exists():
            logger.debug(f"Embeddings file not found: {path}")
            return {}, {}

        try:
            data = np.load(path)
            clip_dict: dict = {}
            t5_dict: dict = {}

            for key in data.files:
                if key.startswith("_"):
                    continue

                if key.endswith("/clip"):
                    base_key = key[:-5]  # strip "/clip"
                    clip_dict[base_key] = data[key]
                elif key.endswith("/t5"):
                    base_key = key[:-3]  # strip "/t5"
                    t5_dict[base_key] = data[key]
                else:
                    # Legacy single-vector format — treat as CLIP
                    clip_dict[key] = data[key]

            return clip_dict, t5_dict
        except Exception as e:
            logger.warning(f"Failed to load embeddings from {path}: {e}")
            return {}, {}
    
    def _initialize_random_state(self) -> None:
        """Initialize with random template and components"""
        if not self.templates:
            logger.error("No templates loaded!")
            return
        
        # Pick random template
        self.current_template = random.choice(list(self.templates.values()))
        
        # Pick random components for each required category
        self._select_components_for_template(self.current_template)
        
        # Generate initial prompt
        self._regenerate_prompt()
        
        logger.info(f"Initialized with template '{self.current_template.id}'")
    
    def _select_components_for_template(self, template: Template) -> None:
        """
        Select random components for all slots in a template
        
        Handles multi-component slots by selecting N different components
        without replacement from the category pool.
        """
        self.current_components = {}
        self.current_multi_components = {}
        
        # Track how many we need from each category (sum of all slots)
        category_needs: Dict[str, int] = {}
        for category, count, _ in template.slots:
            category_needs[category] = category_needs.get(category, 0) + count
        
        # Pre-select all needed components for each category
        for category, total_needed in category_needs.items():
            if category not in self.components:
                logger.warning(f"Category '{category}' not in components pool")
                continue
            
            pool = self.components[category]
            available = min(total_needed, len(pool))
            
            if available < total_needed:
                logger.warning(
                    f"Category '{category}' has {len(pool)} items "
                    f"but template needs {total_needed}"
                )
            
            # Select without replacement
            selected = random.sample(pool, available)
            self.current_multi_components[category] = selected
            
            # First one becomes the "primary" for backwards compatibility
            if selected:
                self.current_components[category] = selected[0]
    
    def _regenerate_prompt(self) -> None:
        """Generate prompt string from current template and components"""
        if not self.current_template:
            self.current_prompt = ""
            self.current_negative = ""
            return
        
        # Build prompt by filling slots
        prompt = self.current_template.structure
        
        # Track consumption index per category (for multi-component slots)
        category_index: Dict[str, int] = {}
        
        # Process each slot
        for category, count, separator in self.current_template.slots:
            if category not in self.current_multi_components:
                continue
            
            # Get starting index for this category
            start_idx = category_index.get(category, 0)
            available = self.current_multi_components[category]
            
            # Get the components for this slot
            end_idx = min(start_idx + count, len(available))
            slot_components = available[start_idx:end_idx]
            
            # Update consumption index
            category_index[category] = end_idx
            
            # Build replacement string
            if slot_components:
                replacement = separator.join(c.word for c in slot_components)
            else:
                replacement = f"[missing {category}]"
            
            # Replace first occurrence of this slot pattern
            # Build the exact pattern we're replacing
            if count > 1:
                if separator != " ":
                    pattern = f"{{{category}:{count}:{separator}}}"
                else:
                    pattern = f"{{{category}:{count}}}"
            else:
                pattern = f"{{{category}}}"
            
            prompt = prompt.replace(pattern, replacement, 1)
        
        self.current_prompt = prompt
        
        # Generate negative prompt from semantic opposites
        self._regenerate_negative()
    
    def _regenerate_negative(self) -> None:
        """Generate negative prompt from semantic opposites of negatable categories"""
        opposites = []
        
        for category in self.NEGATABLE_CATEGORIES:
            if category not in self.current_multi_components:
                continue
            
            for component in self.current_multi_components[category]:
                if component.opposite:
                    opposites.append(component.opposite)
        
        # Combine with base quality negatives
        base_negatives = [
            "low quality",
            "blurry", 
            "text",
            "watermark"
        ]
        
        self.current_negative = ", ".join(opposites + base_negatives)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC INTERFACE - Compatible with existing PromptManager
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_next_prompt(self) -> str:
        """
        Get next prompt (may trigger automatic mutation)
        
        This is the main interface used by KeyframeWorker.
        Increments frame counter and returns current prompt.
        
        Returns:
            Current positive prompt string
        """
        self.total_frames += 1
        self.frames_since_mutation += 1
        
        # Tick bend mode if active
        if self.in_bend_mode:
            self.bend_frames_remaining -= 1
            if self.bend_frames_remaining <= 0:
                self.in_bend_mode = False
                logger.info(
                    f"[STATE] BEND → DRIFT (frame={self.total_frames}, "
                    f"mutations={self.total_mutations})"
                )
            else:
                logger.debug(
                    f"[STATE] BEND mode active, {self.bend_frames_remaining} frames remaining"
                )
        
        return self.current_prompt or ""
    
    def get_negative_prompt(self) -> str:
        """
        Get negative prompt corresponding to current state
        
        Returns:
            Current negative prompt string
        """
        return self.current_negative or ""
    
    def get_current_prompt(self) -> str:
        """
        Get current prompt without incrementing counters
        
        Returns:
            Current positive prompt string
        """
        return self.current_prompt or ""
    
    def get_current_negative(self) -> str:
        """
        Get current negative prompt without incrementing counters
        
        Returns:
            Current negative prompt string
        """
        return self.current_negative or ""
    
    def get_random_prompt(self) -> str:
        """
        Get completely random prompt (new template + all new components)
        
        Useful for cache injection or emergency variety.
        Does NOT update current state.
        
        Returns:
            Random positive prompt string
        """
        # Pick random template
        template = random.choice(list(self.templates.values()))
        
        # Temporarily build prompt without affecting state
        category_pools: Dict[str, List[Component]] = {}
        category_index: Dict[str, int] = {}
        
        # Pre-select components for each category
        for category, count, _ in template.slots:
            if category not in self.components:
                continue
            
            total_needed = category_pools.get(category, [])
            if not total_needed:
                pool = self.components[category]
                # Sum total needed across all slots
                total = sum(c for cat, c, _ in template.slots if cat == category)
                category_pools[category] = random.sample(pool, min(total, len(pool)))
        
        # Fill template
        prompt = template.structure
        
        for category, count, separator in template.slots:
            if category not in category_pools:
                continue
            
            start_idx = category_index.get(category, 0)
            available = category_pools[category]
            end_idx = min(start_idx + count, len(available))
            slot_components = available[start_idx:end_idx]
            category_index[category] = end_idx
            
            if slot_components:
                replacement = separator.join(c.word for c in slot_components)
            else:
                replacement = f"[missing {category}]"
            
            if count > 1:
                if separator != " ":
                    pattern = f"{{{category}:{count}:{separator}}}"
                else:
                    pattern = f"{{{category}:{count}}}"
            else:
                pattern = f"{{{category}}}"
            
            prompt = prompt.replace(pattern, replacement, 1)
        
        return prompt
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MUTATION SYSTEM
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Relative weight of CLIP vs T5 when combining dual-space similarities.
    # CLIP captures visual similarity, T5 captures semantic similarity.
    CLIP_SIMILARITY_WEIGHT = 0.6
    T5_SIMILARITY_WEIGHT = 0.4

    def _select_similar_component(
        self,
        current: Component,
        pool: List[Component]
    ) -> Component:
        """
        Select a new component using dual-space similarity-guided selection

        Computes cosine similarity in both CLIP (visual) and T5 (semantic) spaces,
        then combines them as a weighted average. Prefers components with moderate
        combined similarity — not too similar (boring) and not too different (jarring).

        Falls back gracefully:
        - Both spaces available → weighted combination
        - Only CLIP available → CLIP only
        - Neither available → random selection

        Args:
            current: Current component to transition from
            pool: Available components to select from (excluding current)

        Returns:
            Selected component
        """
        import numpy as np

        has_clip = bool(current.clip_embedding) and len(current.clip_embedding) >= 16
        has_t5 = bool(current.t5_embedding) and len(current.t5_embedding) >= 16

        if not has_clip and not has_t5:
            return random.choice(pool)

        current_clip = np.array(current.clip_embedding) if has_clip else None
        current_t5 = np.array(current.t5_embedding) if has_t5 else None

        # Compute combined similarities to all candidates
        candidates: List[Tuple[Component, float]] = []
        min_sim = self.similarity_target - self.similarity_range
        max_sim = self.similarity_target + self.similarity_range

        for comp in pool:
            comp_has_clip = bool(comp.clip_embedding) and len(comp.clip_embedding) >= 16
            comp_has_t5 = bool(comp.t5_embedding) and len(comp.t5_embedding) >= 16

            # Need at least one shared space
            if not (comp_has_clip or comp_has_t5):
                continue

            sims = []
            weights_for_avg = []

            if has_clip and comp_has_clip:
                clip_sim = float(np.dot(current_clip, np.array(comp.clip_embedding)))
                sims.append(clip_sim)
                weights_for_avg.append(self.CLIP_SIMILARITY_WEIGHT)

            if has_t5 and comp_has_t5:
                t5_sim = float(np.dot(current_t5, np.array(comp.t5_embedding)))
                sims.append(t5_sim)
                weights_for_avg.append(self.T5_SIMILARITY_WEIGHT)

            if not sims:
                continue

            # Weighted average of available similarities
            combined_sim = sum(s * w for s, w in zip(sims, weights_for_avg)) / sum(weights_for_avg)

            if min_sim < combined_sim < max_sim:
                candidates.append((comp, combined_sim))

        if not candidates:
            selected = random.choice(pool)
            logger.info(
                f"[SIMILARITY] No candidates in range [{min_sim:.2f}, {max_sim:.2f}] "
                f"- random fallback: '{selected.word}'"
            )
            return selected

        # Weight candidates: prefer those closer to target similarity
        sel_weights = [
            1.0 - abs(sim - self.similarity_target)
            for _, sim in candidates
        ]

        total = sum(sel_weights)
        if total > 0:
            sel_weights = [w / total for w in sel_weights]
        else:
            sel_weights = [1.0 / len(candidates)] * len(candidates)

        selected = random.choices(candidates, weights=sel_weights, k=1)[0]

        spaces_used = ("CLIP+T5" if (has_clip and has_t5) else
                       "CLIP" if has_clip else "T5")
        top_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]
        logger.info(
            f"[SIMILARITY] '{current.word}' → '{selected[0].word}' "
            f"(sim={selected[1]:.3f}, target={self.similarity_target:.2f}, "
            f"candidates={len(candidates)}/{len(pool)}, space={spaces_used})"
        )
        logger.debug(
            f"[SIMILARITY] Top candidates: {[(c.word, f'{s:.3f}') for c, s in top_candidates]}"
        )

        return selected[0]
    
    def should_mutate(self) -> bool:
        """
        Check if should mutate components
        
        Based on:
        - Random probability
        - Staleness threshold (force mutation after N frames)
        
        Returns:
            True if should mutate
        """
        # Force mutation if stale
        if self.frames_since_mutation >= self.staleness_threshold:
            logger.info(
                f"[MUTATION_CHECK] STALE - forcing mutation "
                f"(frames_since={self.frames_since_mutation}, threshold={self.staleness_threshold})"
            )
            return True
        
        # Random probability
        roll = random.random()
        should = roll < self.mutation_probability
        
        logger.debug(
            f"[MUTATION_CHECK] roll={roll:.3f} vs prob={self.mutation_probability:.3f} "
            f"→ {'MUTATE' if should else 'SKIP'} "
            f"(frames_since={self.frames_since_mutation})"
        )
        
        return should
    
    def mutate(self) -> str:
        """
        Mutate one component category based on weighted selection
        
        - Selects category based on weights
        - Picks new component (different from current)
        - Enters BEND mode for high-denoise frames
        
        Returns:
            New prompt after mutation
        """
        if not self.current_template:
            return self.current_prompt or ""
        
        # Select category to mutate (weighted)
        categories = list(self.category_weights.keys())
        weights = [self.category_weights[c] for c in categories]
        
        # Filter to categories that exist in current template
        available_categories = [
            c for c in categories 
            if c in self.current_multi_components and self.current_multi_components[c]
        ]
        
        if not available_categories:
            logger.warning("No mutable categories in current template")
            return self.current_prompt or ""
        
        # Recalculate weights for available categories
        available_weights = [self.category_weights.get(c, 0.1) for c in available_categories]
        total_weight = sum(available_weights)
        available_weights = [w / total_weight for w in available_weights]
        
        # Select category
        selected_category = random.choices(available_categories, weights=available_weights, k=1)[0]
        
        # Get current component(s) in this category
        current_in_category = self.current_multi_components[selected_category]
        current_words = {c.word for c in current_in_category}
        
        # Get pool excluding current selections
        pool = [c for c in self.components[selected_category] if c.word not in current_words]
        
        if not pool:
            logger.debug(f"No alternative components for '{selected_category}'")
            return self.current_prompt or ""
        
        # Select new component using similarity-guided selection
        # Prefers moderate similarity for smooth visual transitions
        current_component = current_in_category[0]
        new_component = self._select_similar_component(current_component, pool)
        
        # Replace first component in the category (simplest mutation)
        old_component = current_in_category[0]
        current_in_category[0] = new_component
        
        # Update primary component reference
        self.current_components[selected_category] = new_component
        
        # Regenerate prompt
        self._regenerate_prompt()
        
        # Enter BEND mode
        self.in_bend_mode = True
        self.bend_frames_remaining = self.bend_duration
        
        # Update stats
        self.frames_since_mutation = 0
        self.total_mutations += 1
        
        logger.info(
            f"[MUTATE] {selected_category}: '{old_component.word}' → '{new_component.word}' "
            f"| mutation #{self.total_mutations} | DRIFT → BEND ({self.bend_duration}f)"
        )
        logger.info(
            f"[MUTATE] New prompt: {(self.current_prompt or '')[:100]}..."
        )
        
        return self.current_prompt or ""
    
    def is_in_bend_mode(self) -> bool:
        """
        Check if currently in BEND mode (high denoise after mutation)
        
        Returns:
            True if in BEND mode
        """
        return self.in_bend_mode
    
    def force_mutation_opposites(self, n_components: int = 2) -> str:
        """
        Force mutation of N components to their semantic opposites.
        
        Called by collapse detection to push aesthetics away from convergence.
        Uses category weights to select which components to mutate.
        Prioritizes components that have semantic opposites defined.
        Enters BEND mode after mutation.
        
        This is the "soft intervention" for collapse prevention - cheaper than
        cache injection and uses the existing BEND mode infrastructure.
        
        Args:
            n_components: Number of components to replace (default 2)
        
        Returns:
            New prompt after mutation
        """
        if not self.current_template:
            return self.current_prompt or ""
        
        import numpy as np
        
        mutated_categories = []
        
        for _ in range(n_components):
            # Select category based on weights (excluding already mutated this call)
            available_categories = [
                c for c in self.category_weights.keys()
                if c in self.current_multi_components 
                and c not in mutated_categories
                and self.current_multi_components[c]
            ]
            
            if not available_categories:
                break
            
            # Weighted selection
            weights = [self.category_weights.get(c, 0.1) for c in available_categories]
            total = sum(weights)
            weights = [w / total for w in weights]
            
            selected_category = random.choices(available_categories, weights=weights, k=1)[0]
            mutated_categories.append(selected_category)
            
            # Get current component
            current_component = self.current_multi_components[selected_category][0]
            
            # Get pool excluding current
            pool = self.components[selected_category]
            pool_excluding_current = [c for c in pool if c.word != current_component.word]
            
            if not pool_excluding_current:
                continue
            
            # Try to find the ACTUAL semantic opposite of the current component
            # Component.opposite stores the word of its opposite (e.g., "warm" has opposite "cold")
            # So we look for a component whose word matches current_component.opposite
            new_component = None
            
            if current_component.opposite:
                # Look for the component that IS the semantic opposite
                for c in pool_excluding_current:
                    if c.word == current_component.opposite:
                        new_component = c
                        logger.debug(
                            f"[FORCE_MUTATE] Found exact opposite: "
                            f"'{current_component.word}' → '{c.word}'"
                        )
                        break
            
            if new_component is None:
                # No exact opposite found (or current has no opposite defined)
                # Fallback: prefer components that have opposites (more semantically rich)
                candidates_with_opposite = [c for c in pool_excluding_current if c.opposite]
                if candidates_with_opposite:
                    new_component = random.choice(candidates_with_opposite)
                else:
                    # Last resort: any different component
                    new_component = random.choice(pool_excluding_current)
            
            # Apply mutation
            old_word = current_component.word
            self.current_multi_components[selected_category][0] = new_component
            self.current_components[selected_category] = new_component
            
            logger.info(
                f"[FORCE_MUTATE] {selected_category}: '{old_word}' → '{new_component.word}' "
                f"(opposite: {new_component.opposite or 'none'})"
            )
        
        if not mutated_categories:
            logger.warning("[FORCE_MUTATE] No categories available to mutate")
            return self.current_prompt or ""
        
        # Regenerate prompt
        self._regenerate_prompt()
        
        # Enter BEND mode
        self.in_bend_mode = True
        self.bend_frames_remaining = self.bend_duration
        
        # Update stats
        self.frames_since_mutation = 0
        self.total_mutations += len(mutated_categories)
        
        logger.info(
            f"[FORCE_MUTATE] Replaced {len(mutated_categories)} components, "
            f"entering BEND mode ({self.bend_duration}f)"
        )
        logger.info(f"[FORCE_MUTATE] New prompt: {(self.current_prompt or '')[:100]}...")
        
        return self.current_prompt or ""
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPLATE MANAGEMENT (for Phase 5 - Fresh Frame injection)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def switch_template(
        self,
        template_id: Optional[str] = None,
        components: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Switch to a different template
        
        Called at seed injection boundary for template transitions.
        
        Args:
            template_id: ID of template to switch to (random if None)
            components: Optional dict of {category: word} to preset components
        """
        # Select template
        if template_id and template_id in self.templates:
            new_template = self.templates[template_id]
        else:
            # Pick different template than current
            available = [t for t in self.templates.values() if t.id != self.current_template.id]
            if not available:
                available = list(self.templates.values())
            new_template = random.choice(available)
        
        old_template_id = self.current_template.id if self.current_template else "none"
        self.current_template = new_template
        
        # Select new components
        if components:
            # Use provided components where available
            self._select_components_for_template(new_template)
            for category, word in components.items():
                if category in self.components:
                    for comp in self.components[category]:
                        if comp.word == word:
                            self.current_components[category] = comp
                            if category in self.current_multi_components:
                                self.current_multi_components[category][0] = comp
                            break
        else:
            self._select_components_for_template(new_template)
        
        # Regenerate prompt
        self._regenerate_prompt()
        
        # Reset mutation tracking
        self.frames_since_mutation = 0
        self.in_bend_mode = False
        self.bend_frames_remaining = 0
        
        logger.info(f"[TEMPLATE_SWITCH] '{old_template_id}' → '{new_template.id}'")
    
    def get_current_template_id(self) -> str:
        """Get current template ID"""
        return self.current_template.id if self.current_template else ""
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template IDs"""
        return list(self.templates.keys())
    
    def get_current_components(self) -> Dict[str, str]:
        """
        Get current component selections
        
        Returns:
            Dict mapping category to selected word
        """
        return {
            category: comp.word 
            for category, comp in self.current_components.items()
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EMBEDDINGS (for Phase 6 - Similarity calculations)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_prompt_embedding(self, space: str = "clip") -> Optional[List[float]]:
        """
        Get combined embedding for current prompt

        Averages CLIP or T5 embeddings of all current components.
        CLIP embeddings capture visual similarity, T5 captures semantic.

        Args:
            space: Which embedding space — "clip" (default) or "t5"

        Returns:
            Combined embedding vector or None if not available
        """
        embeddings = []

        for components in self.current_multi_components.values():
            for comp in components:
                emb = comp.clip_embedding if space == "clip" else comp.t5_embedding
                if emb:
                    embeddings.append(emb)

        if not embeddings:
            return None

        import numpy as np
        return np.mean(embeddings, axis=0).tolist()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get prompt system statistics
        
        Returns:
            Dictionary with current state and stats
        """
        return {
            "current_template": self.current_template.id if self.current_template else None,
            "current_components": self.get_current_components(),
            "total_frames": self.total_frames,
            "total_mutations": self.total_mutations,
            "frames_since_mutation": self.frames_since_mutation,
            "in_bend_mode": self.in_bend_mode,
            "bend_frames_remaining": self.bend_frames_remaining,
            "templates_available": len(self.templates),
            "mutation_probability": self.mutation_probability
        }
    
    def reset_rotation(self) -> None:
        """
        Reset state (compatibility with PromptManager interface)
        
        Picks new random template and components.
        """
        self._initialize_random_state()
        self.total_frames = 0
        self.frames_since_mutation = 0
        logger.info("Prompt system reset")


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )
    
    print("=" * 70)
    print("Testing CombinatorialPromptSystem")
    print("=" * 70)
    
    # Initialize
    prompt_system = CombinatorialPromptSystem()
    
    # Show initial state
    print(f"\n1. Initial state:")
    print(f"   Template: {prompt_system.get_current_template_id()}")
    print(f"   Components: {prompt_system.get_current_components()}")
    print(f"\n   Positive: {prompt_system.get_current_prompt()}")
    print(f"   Negative: {prompt_system.get_current_negative()}")
    
    # Generate a few prompts
    print(f"\n2. Generating prompts (simulating keyframes):")
    for i in range(5):
        prompt = prompt_system.get_next_prompt()
        negative = prompt_system.get_negative_prompt()
        stats = prompt_system.get_stats()
        print(f"   Frame {i+1}: (bend={stats['in_bend_mode']}) {prompt[:80]}...")
    
    # Test mutation
    print(f"\n3. Forcing mutation:")
    old_prompt = prompt_system.get_current_prompt()
    new_prompt = prompt_system.mutate()
    print(f"   Old: {old_prompt[:80]}...")
    print(f"   New: {new_prompt[:80]}...")
    print(f"   In BEND mode: {prompt_system.is_in_bend_mode()}")
    
    # Test random prompt
    print(f"\n4. Random prompt (doesn't affect state):")
    random_prompt = prompt_system.get_random_prompt()
    print(f"   Random: {random_prompt[:80]}...")
    print(f"   Current (unchanged): {prompt_system.get_current_prompt()[:80]}...")
    
    # Test template switch
    print(f"\n5. Template switch:")
    print(f"   Before: {prompt_system.get_current_template_id()}")
    prompt_system.switch_template()
    print(f"   After: {prompt_system.get_current_template_id()}")
    print(f"   New prompt: {prompt_system.get_current_prompt()[:80]}...")
    
    # Show all templates
    print(f"\n6. Available templates:")
    for template_id in prompt_system.get_available_templates():
        template = prompt_system.templates[template_id]
        print(f"   - {template_id}: {template.structure[:60]}...")
    
    # Generate several random prompts to show variety
    print(f"\n7. Variety test (10 random prompts):")
    for i in range(10):
        prompt = prompt_system.get_random_prompt()
        print(f"   {i+1}. {prompt[:90]}...")
    
    # Final stats
    print(f"\n8. Final statistics:")
    stats = prompt_system.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)

