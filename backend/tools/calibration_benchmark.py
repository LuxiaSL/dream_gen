#!/usr/bin/env python3
"""
Calibration Benchmark Suite
===========================

Proper experimental framework for establishing similarity baselines and
measuring intervention effects. Uses actual dream_gen components.

FRAME-BASED THINKING
--------------------
All measurements are in KEYFRAMES, not time. A "sprint" is 500 keyframes.

With 15 interpolation frames @ 5 FPS:
- 500 keyframes = 7500 total frames = 25 minutes of playback
- But generation is ~3s/keyframe = 25 minutes of generation time

OPTIMIZATION TARGETS
--------------------
For healthy visual variety, we want:

| Intervention     | Target Interval | Reasoning                           |
|------------------|-----------------|-------------------------------------|
| Template switch  | 100 keyframes   | Major visual reset every 3-5 min    |
| Cache injection  | 33-50 keyframes | Medium intervention every 1-2 min   |
| Mutation         | 7-18 keyframes  | Subtle changes every 20-60 seconds  |

These targets inform threshold tuning:
- If mutations fire too rarely → lower mutation_probability or staleness_threshold
- If cache never injects → fix dissimilarity_range to match actual distribution
- If template switches never happen → lower convergence thresholds

Experimental Modes:
-------------------

1. BROAD (Template Survey) - 500 frames across all templates
   - Measures: per-template similarity ranges, inter-template distances
   
2. DEEP (Drift Analysis) - 500 frames on single template
   - Measures: convergence rate, time-to-collapse, mutation effectiveness

3. INTERVENTION (Effect Sizes) - Controlled before/after experiments
   - Measures: actual ΔColorHist and ΔpHash for each intervention type

4. ANALYZE - Process existing keyframes
   - Measures: everything, plus "would config X achieve target Y?"

Usage:
------
    # 500-frame sprint across all templates
    uv run python backend/tools/calibration_benchmark.py broad --frames 500
    
    # 500-frame deep dive on one template
    uv run python backend/tools/calibration_benchmark.py deep --frames 500 --template material_study
    
    # Analyze existing frames with target evaluation
    uv run python backend/tools/calibration_benchmark.py analyze --frames /path/to/keyframes
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import statistics

import numpy as np
from PIL import Image
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import actual dream_gen components
from utils.color_encoder import ColorHistogramEncoder
from utils.phash_encoder import PHashEncoder
from prompts.combinatorial import CombinatorialPromptSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =========================================================================
# OPTIMIZATION TARGETS
# =========================================================================

OPTIMIZATION_TARGETS = {
    'template_switch': {
        'target_interval': 100,  # keyframes between template switches
        'min_acceptable': 75,
        'max_acceptable': 150,
        'description': 'Major visual reset (3-5 min equivalent)'
    },
    'cache_injection': {
        'target_interval': 40,  # keyframes between cache injections
        'min_acceptable': 33,
        'max_acceptable': 50,
        'description': 'Medium intervention (1-2 min equivalent)'
    },
    'mutation': {
        'target_interval': 12,  # keyframes between mutations
        'min_acceptable': 7,
        'max_acceptable': 18,
        'description': 'Subtle changes (20-60 sec equivalent)'
    }
}


class ExperimentMode(Enum):
    BROAD = "broad"           # Template survey
    DEEP = "deep"             # Single template drift
    INTERVENTION = "intervention"  # Effect size measurement
    DIAGONAL = "diagonal"     # Cross-cutting experiments
    ANALYZE = "analyze"       # Analyze existing frames
    FULL = "full"             # Run all modes


@dataclass
class FrameRecord:
    """Complete record for a generated frame"""
    frame_id: int
    timestamp: float
    template_id: str
    prompt: str
    
    # Embeddings (stored separately for efficiency)
    color_embedding: Optional[np.ndarray] = None
    phash_embedding: Optional[object] = None
    
    # Context
    experiment_mode: str = ""
    intervention_type: Optional[str] = None  # mutation_color, mutation_texture, cache_injection, template_switch
    intervention_category: Optional[str] = None
    frames_since_intervention: int = 0
    
    # Computed metrics (filled in post-hoc)
    color_sim_to_prev: Optional[float] = None
    phash_sim_to_prev: Optional[float] = None


@dataclass
class SimilarityDistribution:
    """Statistical distribution of similarity values"""
    values: List[float] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.values)
    
    @property
    def mean(self) -> float:
        return round(statistics.mean(self.values), 4) if self.values else 0.0
    
    @property
    def stdev(self) -> float:
        return round(statistics.stdev(self.values), 4) if len(self.values) > 1 else 0.0
    
    @property
    def median(self) -> float:
        return round(statistics.median(self.values), 4) if self.values else 0.0
    
    def percentile(self, p: int) -> float:
        return round(float(np.percentile(self.values, p)), 4) if self.values else 0.0
    
    def to_dict(self) -> dict:
        if not self.values:
            return {'count': 0}
        return {
            'count': self.count,
            'min': round(min(self.values), 4),
            'max': round(max(self.values), 4),
            'mean': self.mean,
            'median': self.median,
            'stdev': self.stdev,
            'p10': self.percentile(10),
            'p25': self.percentile(25),
            'p75': self.percentile(75),
            'p90': self.percentile(90),
        }


@dataclass
class ExperimentResults:
    """Results from a calibration experiment"""
    mode: str
    start_time: str
    end_time: str
    duration_minutes: float
    total_frames: int
    
    # Raw distributions
    color_all: SimilarityDistribution = field(default_factory=SimilarityDistribution)
    color_consecutive: SimilarityDistribution = field(default_factory=SimilarityDistribution)
    phash_all: SimilarityDistribution = field(default_factory=SimilarityDistribution)
    phash_consecutive: SimilarityDistribution = field(default_factory=SimilarityDistribution)
    
    # Per-template stats
    template_stats: Dict[str, Dict] = field(default_factory=dict)
    
    # Intervention effects (for INTERVENTION mode)
    intervention_effects: Dict[str, Dict] = field(default_factory=dict)
    
    # Time series data (for DEEP mode)
    time_series: List[Dict] = field(default_factory=list)
    
    # Recommendations
    recommended_config: Dict = field(default_factory=dict)
    
    # Optimization target analysis
    target_analysis: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'mode': self.mode,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_minutes': self.duration_minutes,
            'total_frames': self.total_frames,
            'optimization_targets': OPTIMIZATION_TARGETS,
            'color_all': self.color_all.to_dict(),
            'color_consecutive': self.color_consecutive.to_dict(),
            'phash_all': self.phash_all.to_dict(),
            'phash_consecutive': self.phash_consecutive.to_dict(),
            'template_stats': self.template_stats,
            'intervention_effects': self.intervention_effects,
            'time_series': self.time_series,
            'recommended_config': self.recommended_config,
            'target_analysis': self.target_analysis,
        }


class CalibrationBenchmark:
    """
    Calibration benchmark suite using actual dream_gen components
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        output_dir: Path = Path("/workspace/calibration"),
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Initialize actual dream_gen components
        self._init_components()
        
        # Frame storage
        self.frames: List[FrameRecord] = []
        
        # ComfyUI client (lazy init)
        self._comfy_client = None
        
    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration, searching common paths"""
        search_paths = [
            config_path,
            Path(__file__).parent.parent / "config.pod.yaml",
            Path(__file__).parent.parent / "config.yaml",
            Path("/workspace/dream_gen/backend/config.pod.yaml"),
        ]
        
        for p in search_paths:
            if p and p.exists():
                logger.info(f"Loading config from {p}")
                with open(p) as f:
                    return yaml.safe_load(f)
        
        logger.warning("No config found, using defaults")
        return self._default_config()
    
    def _default_config(self) -> dict:
        """Minimal default config for benchmarking"""
        return {
            'system': {
                'comfyui_url': 'http://127.0.0.1:8188',
            },
            'generation': {
                'pipeline': {
                    'width': 512,
                    'height': 288,
                    'steps': 4,
                    'cfg': 1.0,
                },
                'cache': {
                    'color_histogram': {'bins_per_channel': 32},
                    'phash': {'hash_size': 8},
                }
            },
            'prompts': {
                'negative': 'blurry, low quality, watermark, text',
            }
        }
    
    def _init_components(self):
        """Initialize actual dream_gen components"""
        cache_config = self.config.get('generation', {}).get('cache', {})
        color_config = cache_config.get('color_histogram', {})
        phash_config = cache_config.get('phash', {})
        
        # Actual encoders from dream_gen
        self.color_encoder = ColorHistogramEncoder(
            bins_per_channel=color_config.get('bins_per_channel', 32)
        )
        self.phash_encoder = PHashEncoder(
            hash_size=phash_config.get('hash_size', 8)
        )
        
        # Actual prompt system from dream_gen
        # Note: CombinatorialPromptSystem finds templates/components relative to project root
        try:
            self.prompt_system = CombinatorialPromptSystem(config=self.config)
            logger.info(f"Initialized CombinatorialPromptSystem with {len(self.prompt_system.templates)} templates")
            logger.info(f"Templates: {list(self.prompt_system.templates.keys())}")
        except Exception as e:
            logger.warning(f"Could not initialize CombinatorialPromptSystem: {e}")
            self.prompt_system = None
    
    async def _get_comfy_client(self):
        """Lazy-init ComfyUI client"""
        if self._comfy_client is None:
            try:
                from comfy.client import ComfyClient
                comfy_url = self.config.get('system', {}).get('comfyui_url', 'http://127.0.0.1:8188')
                self._comfy_client = ComfyClient(comfy_url)
                await self._comfy_client.connect()
                logger.info(f"Connected to ComfyUI at {comfy_url}")
            except Exception as e:
                logger.error(f"Failed to connect to ComfyUI: {e}")
                raise
        return self._comfy_client
    
    async def _generate_keyframe(
        self,
        prompt: str,
        seed: Optional[int] = None
    ) -> Optional[Image.Image]:
        """Generate a keyframe using actual ComfyUI workflow"""
        try:
            client = await self._get_comfy_client()
            
            from comfy.workflows import build_txt2img_workflow
            
            pipeline = self.config.get('generation', {}).get('pipeline', {})
            
            workflow = build_txt2img_workflow(
                prompt=prompt,
                negative_prompt=self.config.get('prompts', {}).get('negative', ''),
                width=pipeline.get('width', 512),
                height=pipeline.get('height', 288),
                steps=pipeline.get('steps', 4),
                cfg=pipeline.get('cfg', 1.0),
                seed=seed or int(time.time() * 1000) % (2**32),
            )
            
            result = await client.queue_prompt(workflow)
            if result and 'images' in result:
                return result['images'][0]
            return None
            
        except Exception as e:
            logger.error(f"Keyframe generation failed: {e}")
            return None
    
    def _encode_frame(self, image: Image.Image) -> Tuple[np.ndarray, object]:
        """Encode frame using actual dream_gen encoders"""
        color = self.color_encoder.encode_image(image)
        phash = self.phash_encoder.encode_image(image)
        return color, phash
    
    def _compute_similarity(
        self,
        frame_a: FrameRecord,
        frame_b: FrameRecord
    ) -> Tuple[float, float]:
        """Compute similarity using actual dream_gen methods"""
        color_sim = ColorHistogramEncoder.similarity(
            frame_a.color_embedding,
            frame_b.color_embedding
        )
        phash_sim = PHashEncoder.similarity(
            frame_a.phash_embedding,
            frame_b.phash_embedding
        )
        return color_sim, phash_sim
    
    # =========================================================================
    # EXPERIMENT MODES
    # =========================================================================
    
    async def run_broad(self, num_frames: int = 500) -> ExperimentResults:
        """
        BROAD MODE: Template Survey
        
        Generate frames across all templates to understand the full visual span.
        No mutations, no cache - just pure template characteristics.
        
        Args:
            num_frames: Total keyframes to generate (distributed across templates)
        """
        templates = list(self.prompt_system.templates.keys()) if self.prompt_system else ['default']
        frames_per_template = max(1, num_frames // len(templates))
        
        logger.info(f"=== BROAD MODE: Template Survey ({num_frames} frames) ===")
        logger.info(f"  {len(templates)} templates × ~{frames_per_template} frames each")
        
        results = ExperimentResults(
            mode="broad",
            start_time=datetime.now().isoformat(),
            end_time="",
            duration_minutes=0,
            total_frames=0,
        )
        
        start_time = time.time()
        
        frame_id = 0
        template_frames = defaultdict(int)
        
        try:
            while frame_id < num_frames:
                # Pick template with fewest frames (balance coverage)
                template_id = min(templates, key=lambda t: template_frames[t])
                
                if template_frames[template_id] >= frames_per_template:
                    # All templates have enough frames, cycle through
                    if all(template_frames[t] >= frames_per_template for t in templates):
                        template_id = random.choice(templates)
                
                # Generate prompt (fresh random prompt for this template)
                prompt = self.prompt_system.get_random_prompt()
                
                # Generate frame
                logger.info(f"[{frame_id}] Template: {template_id} ({template_frames[template_id]+1}/{frames_per_template})")
                image = await self._generate_keyframe(prompt)
                
                if image is None:
                    continue
                
                # Encode and store
                color_emb, phash_emb = self._encode_frame(image)
                
                record = FrameRecord(
                    frame_id=frame_id,
                    timestamp=time.time(),
                    template_id=template_id,
                    prompt=prompt[:200],
                    color_embedding=color_emb,
                    phash_embedding=phash_emb,
                    experiment_mode="broad",
                )
                self.frames.append(record)
                
                # Save frame
                image.save(self.output_dir / f"broad_{frame_id:04d}_{template_id}.png")
                
                frame_id += 1
                template_frames[template_id] += 1
                
                # Progress
                if frame_id % 50 == 0:
                    elapsed = (time.time() - start_time) / 60
                    logger.info(f"Progress: {frame_id}/{num_frames} frames ({elapsed:.1f} min)")
                
        except KeyboardInterrupt:
            logger.info("Interrupted")
        
        results.end_time = datetime.now().isoformat()
        results.duration_minutes = (time.time() - start_time) / 60
        results.total_frames = len(self.frames)
        
        # Compute statistics
        self._compute_broad_stats(results)
        
        return results
    
    async def run_deep(
        self, 
        num_frames: int = 500, 
        template_id: Optional[str] = None,
        mutation_interval: int = 12  # Target: mutation every 7-18 frames
    ) -> ExperimentResults:
        """
        DEEP MODE: Single Template Drift Analysis
        
        Stay on one template, apply mutations at intervals, measure drift over time.
        
        Args:
            num_frames: Total keyframes to generate
            template_id: Which template to test (random if None)
            mutation_interval: Frames between mutations (default: 12 = target interval)
        """
        templates = list(self.prompt_system.templates.keys()) if self.prompt_system else ['default']
        if template_id is None:
            template_id = random.choice(templates)
        
        logger.info(f"=== DEEP MODE: Drift Analysis on '{template_id}' ({num_frames} frames) ===")
        logger.info(f"  Mutation interval: every {mutation_interval} frames")
        
        results = ExperimentResults(
            mode="deep",
            start_time=datetime.now().isoformat(),
            end_time="",
            duration_minutes=0,
            total_frames=0,
        )
        
        start_time = time.time()
        
        frame_id = 0
        frames_since_mutation = 0
        
        # Get available categories from prompt system
        try:
            # Use actual mutation system
            mutation_count = 0
        except:
            pass
        
        try:
            while frame_id < num_frames:
                # Check if we should mutate
                intervention_type = None
                intervention_category = None
                
                if frames_since_mutation >= mutation_interval and self.prompt_system:
                    # Apply mutation using actual prompt system
                    try:
                        mutated_category = self.prompt_system.mutate()  # Returns category that was mutated
                        intervention_type = "mutation"
                        intervention_category = mutated_category
                        mutation_count += 1
                        frames_since_mutation = 0
                        logger.info(f"[{frame_id}] MUTATION: {mutated_category}")
                    except Exception as e:
                        logger.warning(f"Mutation failed: {e}")
                
                # Generate prompt using actual API
                prompt = self.prompt_system.get_next_prompt() if self.prompt_system else "test prompt"
                
                # Generate frame
                image = await self._generate_keyframe(prompt)
                
                if image is None:
                    continue
                
                # Encode and store
                color_emb, phash_emb = self._encode_frame(image)
                
                record = FrameRecord(
                    frame_id=frame_id,
                    timestamp=time.time(),
                    template_id=template_id,
                    prompt=prompt[:200],
                    color_embedding=color_emb,
                    phash_embedding=phash_emb,
                    experiment_mode="deep",
                    intervention_type=intervention_type,
                    intervention_category=intervention_category,
                    frames_since_intervention=frames_since_mutation,
                )
                
                # Compute similarity to previous frame
                if len(self.frames) > 0:
                    color_sim, phash_sim = self._compute_similarity(self.frames[-1], record)
                    record.color_sim_to_prev = color_sim
                    record.phash_sim_to_prev = phash_sim
                
                self.frames.append(record)
                
                # Save frame
                image.save(self.output_dir / f"deep_{frame_id:04d}.png")
                
                frame_id += 1
                frames_since_mutation += 1
                
                # Progress
                if frame_id % 50 == 0:
                    elapsed = (time.time() - start_time) / 60
                    logger.info(f"Progress: {frame_id}/{num_frames} frames ({elapsed:.1f} min, {mutation_count} mutations)")
                
        except KeyboardInterrupt:
            logger.info("Interrupted")
        
        results.end_time = datetime.now().isoformat()
        results.duration_minutes = (time.time() - start_time) / 60
        results.total_frames = len(self.frames)
        
        # Compute statistics with time series
        self._compute_deep_stats(results)
        
        return results
    
    async def run_intervention(self, num_frames: int = 100) -> ExperimentResults:
        """
        INTERVENTION MODE: Measure Effect Sizes
        
        Controlled experiments to measure the exact effect of each intervention type.
        Structure: baseline (5 frames) → intervention → measure delta → reset → next
        
        Args:
            num_frames: Total keyframes to generate (10 frames per trial = num_frames/10 trials)
        """
        logger.info(f"=== INTERVENTION MODE: Effect Size Measurement ({num_frames} frames) ===")
        
        results = ExperimentResults(
            mode="intervention",
            start_time=datetime.now().isoformat(),
            end_time="",
            duration_minutes=0,
            total_frames=0,
        )
        
        start_time = time.time()
        max_trials = num_frames // 10  # 10 frames per trial (5 baseline + 5 post)
        
        # Intervention types to test
        interventions = [
            ('mutation', 'color_logic'),
            ('mutation', 'medium_render'),
            ('mutation', 'light_behavior'),
            ('mutation', 'texture_density'),
            ('template_switch', None),
        ]
        
        # Results storage
        intervention_data = defaultdict(lambda: {'before': [], 'after': [], 'delta_color': [], 'delta_phash': []})
        
        templates = list(self.prompt_system.templates.keys()) if self.prompt_system else ['default']
        frame_id = 0
        intervention_idx = 0
        trials_completed = 0
        
        try:
            while trials_completed < max_trials:
                intervention_type, category = interventions[intervention_idx % len(interventions)]
                intervention_key = f"{intervention_type}_{category}" if category else intervention_type
                
                logger.info(f"\n--- Testing: {intervention_key} ---")
                
                # Reset to fresh state using random prompt
                template_id = random.choice(templates)
                
                # Generate BASELINE frames (5 frames)
                baseline_frames = []
                for i in range(5):
                    prompt = self.prompt_system.get_next_prompt()
                    image = await self._generate_keyframe(prompt)
                    if image is None:
                        continue
                    
                    color_emb, phash_emb = self._encode_frame(image)
                    baseline_frames.append((color_emb, phash_emb))
                    
                    image.save(self.output_dir / f"intervention_{frame_id:04d}_baseline.png")
                    frame_id += 1
                
                if len(baseline_frames) < 2:
                    continue
                
                # Compute baseline similarity (last two frames)
                baseline_color = ColorHistogramEncoder.similarity(
                    baseline_frames[-2][0], baseline_frames[-1][0]
                )
                baseline_phash = PHashEncoder.similarity(
                    baseline_frames[-2][1], baseline_frames[-1][1]
                )
                
                intervention_data[intervention_key]['before'].append({
                    'color': baseline_color,
                    'phash': baseline_phash
                })
                
                # Apply INTERVENTION
                if intervention_type == 'mutation':
                    # Use force_mutation_opposites for controlled mutation test
                    self.prompt_system.force_mutation_opposites(n_components=1)
                elif intervention_type == 'template_switch':
                    template_id = random.choice([t for t in templates if t != template_id])
                
                # Generate POST-INTERVENTION frames (5 frames)
                post_frames = []
                for i in range(5):
                    prompt = self.prompt_system.get_next_prompt()
                    image = await self._generate_keyframe(prompt)
                    if image is None:
                        continue
                    
                    color_emb, phash_emb = self._encode_frame(image)
                    post_frames.append((color_emb, phash_emb))
                    
                    image.save(self.output_dir / f"intervention_{frame_id:04d}_post.png")
                    frame_id += 1
                
                if len(post_frames) < 1:
                    continue
                
                # Compute cross-intervention similarity (last baseline to first post)
                cross_color = ColorHistogramEncoder.similarity(
                    baseline_frames[-1][0], post_frames[0][0]
                )
                cross_phash = PHashEncoder.similarity(
                    baseline_frames[-1][1], post_frames[0][1]
                )
                
                intervention_data[intervention_key]['after'].append({
                    'color': cross_color,
                    'phash': cross_phash
                })
                
                # Compute delta
                delta_color = cross_color - baseline_color
                delta_phash = cross_phash - baseline_phash
                
                intervention_data[intervention_key]['delta_color'].append(delta_color)
                intervention_data[intervention_key]['delta_phash'].append(delta_phash)
                
                logger.info(f"  Baseline: color={baseline_color:.3f}, phash={baseline_phash:.3f}")
                logger.info(f"  After:    color={cross_color:.3f}, phash={cross_phash:.3f}")
                logger.info(f"  Delta:    color={delta_color:+.3f}, phash={delta_phash:+.3f}")
                
                intervention_idx += 1
                trials_completed += 1
                
                # Progress
                elapsed = (time.time() - start_time) / 60
                logger.info(f"  Progress: {trials_completed}/{max_trials} trials ({frame_id} frames, {elapsed:.1f} min)")
                
        except KeyboardInterrupt:
            logger.info("Interrupted")
        
        results.end_time = datetime.now().isoformat()
        results.duration_minutes = (time.time() - start_time) / 60
        results.total_frames = frame_id
        
        # Summarize intervention effects
        for key, data in intervention_data.items():
            if data['delta_color']:
                results.intervention_effects[key] = {
                    'n_trials': len(data['delta_color']),
                    'color_delta_mean': round(statistics.mean(data['delta_color']), 4),
                    'color_delta_stdev': round(statistics.stdev(data['delta_color']), 4) if len(data['delta_color']) > 1 else 0,
                    'phash_delta_mean': round(statistics.mean(data['delta_phash']), 4),
                    'phash_delta_stdev': round(statistics.stdev(data['delta_phash']), 4) if len(data['delta_phash']) > 1 else 0,
                }
        
        return results
    
    def analyze_existing(self, frame_dir: Path) -> ExperimentResults:
        """
        ANALYZE MODE: Process existing keyframes
        """
        logger.info(f"=== ANALYZE MODE: {frame_dir} ===")
        
        results = ExperimentResults(
            mode="analyze",
            start_time=datetime.now().isoformat(),
            end_time="",
            duration_minutes=0,
            total_frames=0,
        )
        
        # Find frames
        frame_files = sorted(frame_dir.glob("*.png"))
        logger.info(f"Found {len(frame_files)} frames")
        
        for i, frame_path in enumerate(frame_files):
            if i % 50 == 0:
                logger.info(f"Processing {i}/{len(frame_files)}")
            
            image = Image.open(frame_path).convert('RGB')
            color_emb, phash_emb = self._encode_frame(image)
            
            # Try to extract template from filename
            template_id = self._extract_template_from_filename(frame_path.stem)
            
            record = FrameRecord(
                frame_id=i,
                timestamp=frame_path.stat().st_mtime,
                template_id=template_id,
                prompt="",
                color_embedding=color_emb,
                phash_embedding=phash_emb,
                experiment_mode="analyze",
            )
            
            # Compute similarity to previous
            if len(self.frames) > 0:
                color_sim, phash_sim = self._compute_similarity(self.frames[-1], record)
                record.color_sim_to_prev = color_sim
                record.phash_sim_to_prev = phash_sim
            
            self.frames.append(record)
        
        results.end_time = datetime.now().isoformat()
        results.total_frames = len(self.frames)
        
        # Compute full statistics
        self._compute_full_stats(results)
        
        return results
    
    def _extract_template_from_filename(self, stem: str) -> str:
        """Try to extract template ID from filename"""
        # Common patterns: "keyframe_001", "broad_0001_material_study", "deep_0001"
        parts = stem.split("_")
        
        # Check if any part matches a known template
        templates = list(self.prompt_system.templates.keys())
        for part in parts:
            if part in templates:
                return part
        
        # Check for combined parts (e.g., "material_study")
        for i in range(len(parts) - 1):
            combined = f"{parts[i]}_{parts[i+1]}"
            if combined in templates:
                return combined
        
        return "unknown"
    
    # =========================================================================
    # STATISTICS COMPUTATION
    # =========================================================================
    
    def _compute_broad_stats(self, results: ExperimentResults):
        """Compute statistics for BROAD mode (per-template focus)"""
        # Group frames by template
        template_frames = defaultdict(list)
        for frame in self.frames:
            template_frames[frame.template_id].append(frame)
        
        # Compute per-template stats
        for template_id, frames in template_frames.items():
            if len(frames) < 2:
                continue
            
            color_sims = []
            phash_sims = []
            
            # Within-template similarities
            for i in range(len(frames)):
                for j in range(i + 1, len(frames)):
                    c, p = self._compute_similarity(frames[i], frames[j])
                    color_sims.append(c)
                    phash_sims.append(p)
            
            results.template_stats[template_id] = {
                'n_frames': len(frames),
                'color': SimilarityDistribution(color_sims).to_dict(),
                'phash': SimilarityDistribution(phash_sims).to_dict(),
            }
        
        # Also compute overall stats
        self._compute_full_stats(results)
    
    def _compute_deep_stats(self, results: ExperimentResults):
        """Compute statistics for DEEP mode (time series focus)"""
        # Build time series
        window_size = 20  # Rolling window
        
        for i in range(window_size, len(self.frames)):
            window = self.frames[i-window_size:i]
            
            # Compute average similarity in window
            color_sims = [f.color_sim_to_prev for f in window if f.color_sim_to_prev is not None]
            phash_sims = [f.phash_sim_to_prev for f in window if f.phash_sim_to_prev is not None]
            
            if color_sims and phash_sims:
                results.time_series.append({
                    'frame_id': i,
                    'timestamp': self.frames[i].timestamp,
                    'color_mean': round(statistics.mean(color_sims), 4),
                    'phash_mean': round(statistics.mean(phash_sims), 4),
                    'intervention': self.frames[i].intervention_type,
                })
        
        # Also compute overall stats
        self._compute_full_stats(results)
    
    def _compute_full_stats(self, results: ExperimentResults):
        """Compute comprehensive statistics from all frames"""
        if len(self.frames) < 2:
            return
        
        # All pairwise similarities
        color_all = []
        phash_all = []
        color_consecutive = []
        phash_consecutive = []
        
        for i in range(len(self.frames)):
            for j in range(i + 1, len(self.frames)):
                c, p = self._compute_similarity(self.frames[i], self.frames[j])
                color_all.append(c)
                phash_all.append(p)
                
                if j == i + 1:
                    color_consecutive.append(c)
                    phash_consecutive.append(p)
        
        results.color_all = SimilarityDistribution(color_all)
        results.phash_all = SimilarityDistribution(phash_all)
        results.color_consecutive = SimilarityDistribution(color_consecutive)
        results.phash_consecutive = SimilarityDistribution(phash_consecutive)
        
        # Generate recommendations
        self._generate_recommendations(results)
    
    def _generate_recommendations(self, results: ExperimentResults):
        """Generate recommended threshold values and target analysis"""
        color = results.color_all
        phash = results.phash_all
        color_consec = results.color_consecutive
        phash_consec = results.phash_consecutive
        
        if color.count == 0 or phash.count == 0:
            return
        
        # === THRESHOLD RECOMMENDATIONS ===
        results.recommended_config = {
            'color_histogram': {
                'diversity_threshold': round(color.percentile(75) + 0.1, 2),
                'dissimilarity_range': [round(color.percentile(25), 2), round(color.percentile(90), 2)],
                'convergence_threshold': round(color.stdev * 1.5, 2),
                'force_cache_threshold': round(color.stdev * 3.0, 2),
            },
            'phash': {
                'diversity_threshold': round(phash.percentile(75) + 0.05, 2),
                'dissimilarity_range': [round(phash.percentile(25), 2), round(phash.percentile(90), 2)],
                'convergence_threshold': round(phash.stdev * 1.2, 2),
                'force_cache_threshold': round(phash.stdev * 2.5, 2),
            }
        }
        
        # === TARGET ANALYSIS ===
        # Analyze what intervention frequency the recommended config would produce
        rec = results.recommended_config
        
        # For mutations: based on staleness_threshold and mutation_probability
        # Default: 12.5% probability, 30 staleness → expected interval ~8-10 frames
        # This is handled by the prompt system, not thresholds
        mutation_expected = "Controlled by mutation_probability (12.5%) and staleness_threshold (30)"
        
        # For cache injection: how often would dissimilarity_range catch a frame?
        # Count what % of consecutive frame similarities fall in the range
        cache_range = rec['phash']['dissimilarity_range']
        if phash_consec.values:
            in_range = sum(1 for v in phash_consec.values if cache_range[0] <= v <= cache_range[1])
            cache_hit_rate = in_range / len(phash_consec.values)
            # Expected interval = 1 / hit_rate (if all frames were candidates)
            cache_expected_interval = round(1 / cache_hit_rate, 1) if cache_hit_rate > 0 else float('inf')
        else:
            cache_hit_rate = 0
            cache_expected_interval = float('inf')
        
        # For template switch: based on convergence detection triggering
        # This requires tracking consecutive similarity trends over time
        # Estimate: if std of consecutive is X, and threshold is 1.5*X, expect trigger every ~50 frames?
        # This is rough - actual behavior depends on rolling window dynamics
        template_switch_note = "Depends on collapse_detection_window and baseline comparison"
        
        results.target_analysis = {
            'optimization_targets': OPTIMIZATION_TARGETS,
            'mutation': {
                'target_interval': OPTIMIZATION_TARGETS['mutation']['target_interval'],
                'expected_behavior': mutation_expected,
                'recommendation': 'Use staleness_threshold=15-20 for target ~12 frame interval'
            },
            'cache_injection': {
                'target_interval': OPTIMIZATION_TARGETS['cache_injection']['target_interval'],
                'recommended_range': cache_range,
                'hit_rate_in_consecutive': round(cache_hit_rate * 100, 1),
                'expected_interval_frames': cache_expected_interval,
                'meets_target': OPTIMIZATION_TARGETS['cache_injection']['min_acceptable'] <= cache_expected_interval <= OPTIMIZATION_TARGETS['cache_injection']['max_acceptable'],
                'recommendation': f"{'✅ Good' if OPTIMIZATION_TARGETS['cache_injection']['min_acceptable'] <= cache_expected_interval <= OPTIMIZATION_TARGETS['cache_injection']['max_acceptable'] else '⚠️ Adjust range'}: {cache_hit_rate*100:.1f}% hit rate → ~{cache_expected_interval:.0f} frame interval"
            },
            'template_switch': {
                'target_interval': OPTIMIZATION_TARGETS['template_switch']['target_interval'],
                'note': template_switch_note,
                'consecutive_phash_stats': {
                    'mean': phash_consec.mean,
                    'stdev': phash_consec.stdev,
                },
                'recommendation': f"convergence_threshold={rec['phash']['convergence_threshold']} should trigger when drift exceeds {rec['phash']['convergence_threshold']}"
            },
            'summary': {
                'total_frames_analyzed': results.total_frames,
                'frames_per_sprint': 500,
                'expected_per_sprint': {
                    'template_switches': round(500 / OPTIMIZATION_TARGETS['template_switch']['target_interval'], 1),
                    'cache_injections': round(500 / OPTIMIZATION_TARGETS['cache_injection']['target_interval'], 1),
                    'mutations': round(500 / OPTIMIZATION_TARGETS['mutation']['target_interval'], 1),
                }
            }
        }
    
    # =========================================================================
    # OUTPUT
    # =========================================================================
    
    def save_results(self, results: ExperimentResults, output_path: Optional[Path] = None):
        """Save results to JSON"""
        if output_path is None:
            output_path = self.output_dir / f"calibration_{results.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        self._print_summary(results)
        
        return output_path
    
    def _print_summary(self, results: ExperimentResults):
        """Print human-readable summary"""
        print("\n" + "=" * 70)
        print(f"CALIBRATION RESULTS: {results.mode.upper()}")
        print("=" * 70)
        
        print(f"\nDuration: {results.duration_minutes:.1f} minutes")
        print(f"Frames: {results.total_frames}")
        
        print("\n--- ColorHist (higher = more different) ---")
        ca = results.color_all.to_dict()
        cc = results.color_consecutive.to_dict()
        if ca.get('count', 0) > 0:
            print(f"  All pairs:    mean={ca['mean']:.3f}, σ={ca['stdev']:.3f}, range=[{ca['min']:.3f}, {ca['max']:.3f}]")
        if cc.get('count', 0) > 0:
            print(f"  Consecutive:  mean={cc['mean']:.3f}, σ={cc['stdev']:.3f}, range=[{cc['min']:.3f}, {cc['max']:.3f}]")
        
        print("\n--- pHash (higher = more similar) ---")
        pa = results.phash_all.to_dict()
        pc = results.phash_consecutive.to_dict()
        if pa.get('count', 0) > 0:
            print(f"  All pairs:    mean={pa['mean']:.3f}, σ={pa['stdev']:.3f}, range=[{pa['min']:.3f}, {pa['max']:.3f}]")
        if pc.get('count', 0) > 0:
            print(f"  Consecutive:  mean={pc['mean']:.3f}, σ={pc['stdev']:.3f}, range=[{pc['min']:.3f}, {pc['max']:.3f}]")
        
        if results.template_stats:
            print(f"\n--- Per-Template Stats ({len(results.template_stats)} templates) ---")
            for tid, stats in sorted(results.template_stats.items()):
                cs = stats.get('color', {})
                ps = stats.get('phash', {})
                print(f"  {tid}: color={cs.get('mean', 0):.3f}±{cs.get('stdev', 0):.3f}, phash={ps.get('mean', 0):.3f}±{ps.get('stdev', 0):.3f}")
        
        if results.intervention_effects:
            print("\n--- Intervention Effects ---")
            for key, effect in sorted(results.intervention_effects.items()):
                print(f"  {key}:")
                print(f"    ColorHist Δ: {effect['color_delta_mean']:+.3f} ± {effect['color_delta_stdev']:.3f}")
                print(f"    pHash Δ:     {effect['phash_delta_mean']:+.3f} ± {effect['phash_delta_stdev']:.3f}")
        
        if results.recommended_config:
            print("\n--- Recommended Config ---")
            rec = results.recommended_config
            print("cache:")
            print("  color_histogram:")
            for k, v in rec.get('color_histogram', {}).items():
                print(f"    {k}: {v}")
            print("  phash:")
            for k, v in rec.get('phash', {}).items():
                print(f"    {k}: {v}")
        
        if results.target_analysis:
            print("\n--- Optimization Target Analysis ---")
            ta = results.target_analysis
            
            print("\nTargets (per 500-keyframe sprint):")
            if 'summary' in ta:
                exp = ta['summary'].get('expected_per_sprint', {})
                print(f"  Template switches: {exp.get('template_switches', '?')} (target: 5)")
                print(f"  Cache injections:  {exp.get('cache_injections', '?')} (target: 10-15)")
                print(f"  Mutations:         {exp.get('mutations', '?')} (target: 28-42)")
            
            print("\nCache Injection Analysis:")
            ci = ta.get('cache_injection', {})
            print(f"  Recommended pHash range: {ci.get('recommended_range', '?')}")
            print(f"  Hit rate in consecutive: {ci.get('hit_rate_in_consecutive', '?')}%")
            print(f"  Expected interval: ~{ci.get('expected_interval_frames', '?')} frames")
            print(f"  {ci.get('recommendation', '')}")
            
            print("\nMutation Analysis:")
            mut = ta.get('mutation', {})
            print(f"  Target interval: {mut.get('target_interval', '?')} frames")
            print(f"  {mut.get('recommendation', '')}")
        
        print("\n" + "=" * 70)


async def main():
    parser = argparse.ArgumentParser(
        description="Calibration Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Frame-Based Benchmarking:
  A "sprint" is 500 keyframes. With 15 interpolation frames @ 5 FPS:
  500 keyframes = 7500 total frames = 25 min playback = ~25 min generation

Examples:
  # 500-frame sprint across all templates
  %(prog)s broad --num-frames 500
  
  # 500-frame deep dive on material_study
  %(prog)s deep --num-frames 500 --template material_study
  
  # Measure intervention effect sizes (100 frames)
  %(prog)s intervention --num-frames 100
  
  # Analyze existing keyframes
  %(prog)s analyze --frames /path/to/keyframes
  
  # Full suite (1500 frames total)
  %(prog)s full --num-frames 1500
"""
    )
    
    parser.add_argument('mode', choices=['broad', 'deep', 'intervention', 'diagonal', 'analyze', 'full'],
                        help='Experiment mode')
    parser.add_argument('--num-frames', type=int, default=500,
                        help='Number of keyframes to generate (default: 500 = 1 sprint)')
    parser.add_argument('--template', type=str, default=None,
                        help='Template ID for deep mode')
    parser.add_argument('--frames', type=str, default=None,
                        help='Frame directory for analyze mode')
    parser.add_argument('--output-dir', type=str, default='/workspace/calibration',
                        help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file path')
    
    args = parser.parse_args()
    
    benchmark = CalibrationBenchmark(
        config_path=Path(args.config) if args.config else None,
        output_dir=Path(args.output_dir),
    )
    
    if args.mode == 'broad':
        results = await benchmark.run_broad(num_frames=args.num_frames)
    elif args.mode == 'deep':
        results = await benchmark.run_deep(num_frames=args.num_frames, template_id=args.template)
    elif args.mode == 'intervention':
        results = await benchmark.run_intervention(num_frames=args.num_frames)
    elif args.mode == 'analyze':
        if not args.frames:
            parser.error("--frames required for analyze mode")
        results = benchmark.analyze_existing(Path(args.frames))
    elif args.mode == 'full':
        # Run all modes sequentially
        logger.info("Running full calibration suite...")
        all_results = []
        
        # Broad first (1/4 of frames)
        results = await benchmark.run_broad(num_frames=args.num_frames // 4)
        benchmark.save_results(results)
        all_results.append(results)
        benchmark.frames.clear()
        
        # Deep (1/2 of frames)
        results = await benchmark.run_deep(num_frames=args.num_frames // 2)
        benchmark.save_results(results)
        all_results.append(results)
        benchmark.frames.clear()
        
        # Intervention (1/4 of frames)
        results = await benchmark.run_intervention(num_frames=args.num_frames // 4)
        benchmark.save_results(results)
        all_results.append(results)
        
        # Combined summary
        total_frames = sum(r.total_frames for r in all_results)
        results = ExperimentResults(
            mode="full", 
            start_time=all_results[0].start_time,
            end_time=all_results[-1].end_time, 
            duration_minutes=sum(r.duration_minutes for r in all_results),
            total_frames=total_frames
        )
    else:
        parser.error(f"Unknown mode: {args.mode}")
        return
    
    benchmark.save_results(results)


if __name__ == "__main__":
    asyncio.run(main())
