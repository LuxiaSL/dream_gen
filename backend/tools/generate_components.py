#!/usr/bin/env python3
"""
Component Generation Tool v2
═══════════════════════════════════════════════════════════════════════════════

Uses Anthropic SDK directly for Claude (with prompt caching) and OpenRouter
for Kimi K2 (with Moonshot provider caching). Alternates between models,
each adding to the cumulative word pool.

Features:
- Anthropic SDK with proper prompt caching (cache reads = 10% of base price)
- OpenRouter + Moonshot provider for Kimi K2 (automatic caching)
- Careful prompt structure: static parts first (cached), dynamic parts last
- 6 parallel requests by default
- Alternating model strategy for diversity

Usage:
    # Set API keys
    export ANTHROPIC_API_KEY="sk-ant-..."
    export OPENROUTER_API_KEY="sk-or-v1-..."
    
    # Generate components
    uv run python backend/tools/generate_components.py
    
    # Generate for specific categories
    uv run python backend/tools/generate_components.py --categories subject_form material_substance
    
    # Dry run (show prompts without calling API)
    uv run python backend/tools/generate_components.py --dry-run
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import aiohttp
import yaml

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# API Endpoints
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Model configurations
CLAUDE_MODEL = "claude-opus-4-5"  # Use direct model ID for Anthropic SDK
KIMI_MODEL = "moonshotai/kimi-k2"  # OpenRouter model ID

# Generation settings
BATCH_SIZE = 20  # Components per request (LLMs handle 20 easily, filtering handles dupes)
TARGET_PER_CATEGORY = 200  # Target components per category
MAX_PARALLEL_REQUESTS = 6  # Concurrent requests
REQUEST_DELAY_MS = 100  # Small delay between batches
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N iterations

# Claude Opus pricing (per million tokens)
CLAUDE_PRICING = {
    "input": 5.00,        # Base input tokens
    "cache_write": 6.25,  # Cache creation tokens (1.25x base)
    "cache_read": 0.50,   # Cache hit tokens (0.1x base)
    "output": 25.00,      # Output tokens
}

# Default cost cap for Claude before switching to Kimi-only
DEFAULT_CLAUDE_COST_CAP = 5.00  # USD

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CategoryConfig:
    """Configuration for a component category"""
    name: str
    description: str
    generation_guidance: str
    anti_examples: list[str]
    seeds: list[str]
    sub_clusters: list[str] = field(default_factory=list)
    target_count: int = 200


@dataclass
class GenerationResult:
    """Result from a single generation batch"""
    category: str
    model: str
    components: list[str]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    # Claude-specific token breakdown
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    # Uncached input tokens (for Claude: input_tokens field from API)
    uncached_input_tokens: int = 0


@dataclass
class GenerationState:
    """Tracks generation progress across categories"""
    components: dict[str, set[str]] = field(default_factory=dict)
    total_tokens: int = 0
    total_cached_tokens: int = 0
    total_requests: int = 0
    errors: list[str] = field(default_factory=list)
    
    # Per-model cost tracking
    claude_cost: float = 0.0
    kimi_cost: float = 0.0  # Kimi cost tracking (for reference, not capped)
    
    # Token breakdown for Claude
    claude_input_tokens: int = 0
    claude_cache_write_tokens: int = 0
    claude_cache_read_tokens: int = 0
    claude_output_tokens: int = 0

    def add_components(self, category: str, new_components: list[str]) -> int:
        """Add components and return count of newly added (unique) ones"""
        if category not in self.components:
            self.components[category] = set()
        before = len(self.components[category])
        self.components[category].update(new_components)
        return len(self.components[category]) - before
    
    def add_claude_usage(
        self,
        input_tokens: int,
        cache_write_tokens: int,
        cache_read_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Add Claude token usage and calculate cost.
        Returns the cost of this request.
        """
        self.claude_input_tokens += input_tokens
        self.claude_cache_write_tokens += cache_write_tokens
        self.claude_cache_read_tokens += cache_read_tokens
        self.claude_output_tokens += output_tokens
        
        # Calculate cost (per million tokens)
        cost = (
            (input_tokens * CLAUDE_PRICING["input"] / 1_000_000) +
            (cache_write_tokens * CLAUDE_PRICING["cache_write"] / 1_000_000) +
            (cache_read_tokens * CLAUDE_PRICING["cache_read"] / 1_000_000) +
            (output_tokens * CLAUDE_PRICING["output"] / 1_000_000)
        )
        
        self.claude_cost += cost
        return cost


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

# Static system prompt - THIS GETS CACHED
# Kept minimal and stable so cache hits are maximized
SYSTEM_PROMPT_STATIC = """You are helping expand a vocabulary pool for an AI image generation system using Stable Diffusion 1.5.

The system uses combinatorial prompts where different categories control independent visual axes:
- Each category should have visually distinct terms
- Terms should be specific enough that CLIP (the text encoder) responds to them consistently
- Avoid overlap between categories - if a term could fit multiple categories, it doesn't belong

Your task is to generate NEW terms for a specific category that are:
1. Visually distinct from the provided examples (they should look different when rendered)
2. Specific and concrete (not vague or abstract)
3. Within the category's scope (follow the guidance, avoid anti-examples)
4. 1-4 words maximum per term

Output ONLY the new terms, one per line. No numbering, no explanations, no commentary."""


def make_category_context(config: CategoryConfig) -> str:
    """
    Build the STATIC category context that will be cached.
    This includes everything except the dynamic "words to avoid" list.
    """
    anti_examples_str = "\n".join(f"  - {ex}" for ex in config.anti_examples)
    seeds_str = "\n".join(f"  - {seed}" for seed in config.seeds)
    
    sub_clusters_str = ""
    if config.sub_clusters:
        sub_clusters_str = "\n\nSub-clusters (generate roughly EQUAL numbers from each):\n"
        sub_clusters_str += "\n".join(f"  - {sc}" for sc in config.sub_clusters)
        sub_clusters_str += "\n\nIMPORTANT: Balance your generation across ALL sub-clusters, not just the first one."
    
    return f"""Category: {config.name.upper().replace('_', '-')}

Description: {config.description}

Generation Guidance: {config.generation_guidance}

Anti-examples (do NOT generate terms like these):
{anti_examples_str}

Seed examples (generate NEW ones visually distinct from these):
{seeds_str}{sub_clusters_str}"""


def make_user_message(existing: set[str], batch_size: int = BATCH_SIZE) -> str:
    """
    Build the DYNAMIC user message - this is NOT cached.
    Contains only the words to avoid (which grows each iteration).
    
    Uses random sampling (not sorted) for better semantic coverage of the avoid space.
    """
    if existing:
        # Random sample gives better coverage than alphabetically-sorted
        existing_list = list(existing)
        sample_size = min(50, len(existing_list))
        avoid_sample = random.sample(existing_list, sample_size)
        avoid_str = ", ".join(avoid_sample)
        if len(existing) > sample_size:
            avoid_str += f" ... and {len(existing) - sample_size} others"
        return f"""AVOID these existing terms: {avoid_str}

Generate {batch_size} new unique terms. Output ONLY the terms, one per line."""
    else:
        return f"Generate {batch_size} new unique terms. Output ONLY the terms, one per line."


# ═══════════════════════════════════════════════════════════════════════════════
# ANTHROPIC CLIENT (Direct SDK with Prompt Caching)
# ═══════════════════════════════════════════════════════════════════════════════


async def call_anthropic(
    session: aiohttp.ClientSession,
    category_context: str,
    user_message: str,
    api_key: str,
) -> dict[str, Any]:
    """
    Call Anthropic API directly with prompt caching.
    
    Structure:
    - System: [Static instructions (cached)] + [Category context (cached)]
    - User: [Dynamic avoid list + request (NOT cached)]
    
    Per Anthropic docs: cache_control marks the END of the cached prefix.
    Everything before and including the marked block is cached.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    
    # Build system message with cache breakpoint
    # The category context is appended to static instructions, all cached together
    full_system = f"{SYSTEM_PROMPT_STATIC}\n\n---\n\n{category_context}"
    
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "temperature": 0.9,
        "system": [
            {
                "type": "text",
                "text": full_system,
                "cache_control": {"type": "ephemeral"},  # 5-min cache
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": user_message,  # Dynamic part - NOT cached
            }
        ],
    }
    
    async with session.post(ANTHROPIC_API_URL, headers=headers, json=payload) as response:
        if response.status != 200:
            error_text = await response.text()
            raise RuntimeError(f"Anthropic API error {response.status}: {error_text}")
        return await response.json()


def parse_anthropic_response(response: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    """
    Parse Anthropic response into components and token counts.
    
    Anthropic usage fields:
    - input_tokens: Uncached input tokens (AFTER last cache breakpoint)
    - cache_creation_input_tokens: Tokens written to cache (first request)
    - cache_read_input_tokens: Tokens read from cache (subsequent requests)
    - output_tokens: Generated output tokens
    
    Total input = cache_read + cache_creation + input_tokens
    """
    # Extract text content
    content = ""
    for block in response.get("content", []):
        if block.get("type") == "text":
            content = block.get("text", "")
            break
    
    # Parse components
    components = parse_component_lines(content)
    
    # Extract token usage - Anthropic's breakdown
    usage = response.get("usage", {})
    
    uncached_input = usage.get("input_tokens", 0)  # Tokens after cache breakpoint
    cache_write = usage.get("cache_creation_input_tokens", 0)  # New cache created
    cache_read = usage.get("cache_read_input_tokens", 0)  # Cache hit
    output = usage.get("output_tokens", 0)
    
    token_info = {
        "uncached_input_tokens": uncached_input,
        "cache_write_tokens": cache_write,
        "cache_read_tokens": cache_read,
        "output_tokens": output,
        # For backwards compat / total tracking
        "prompt_tokens": uncached_input + cache_write + cache_read,
        "completion_tokens": output,
    }
    
    return components, token_info


# ═══════════════════════════════════════════════════════════════════════════════
# OPENROUTER CLIENT (For Kimi K2 with Moonshot provider)
# ═══════════════════════════════════════════════════════════════════════════════


async def call_openrouter_kimi(
    session: aiohttp.ClientSession,
    category_context: str,
    user_message: str,
    api_key: str,
) -> dict[str, Any]:
    """
    Call OpenRouter for Kimi K2, forcing Moonshot provider for automatic caching.
    
    Per OpenRouter docs: Moonshot AI caching is automatic.
    We force the provider to ensure we always hit their cached infrastructure.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/luxia/dream-gen",
        "X-Title": "Dream Gen Component Generator",
    }
    
    full_system = f"{SYSTEM_PROMPT_STATIC}\n\n---\n\n{category_context}"
    
    payload = {
        "model": KIMI_MODEL,
        "messages": [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.9,
        "max_tokens": 1024,
        "provider": {
            "only": ["moonshotai/fp8"],  # Force Moonshot provider for caching
            "allow_fallbacks": False,
        },
    }
    
    async with session.post(OPENROUTER_API_URL, headers=headers, json=payload) as response:
        if response.status != 200:
            error_text = await response.text()
            raise RuntimeError(f"OpenRouter API error {response.status}: {error_text}")
        return await response.json()


def parse_openrouter_response(response: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    """Parse OpenRouter response into components and token counts"""
    content = response["choices"][0]["message"]["content"]
    components = parse_component_lines(content)
    
    usage = response.get("usage", {})
    token_info = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
    }
    
    return components, token_info


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def parse_component_lines(content: str) -> list[str]:
    """Parse raw LLM output into clean component list"""
    raw_lines = content.strip().split("\n")
    components = []
    
    for line in raw_lines:
        # Remove numbering, bullets, quotes
        cleaned = re.sub(r"^[\d\.\-\*\•]+\s*", "", line.strip())
        cleaned = cleaned.strip('"\'')
        cleaned = cleaned.strip()
        
        # Validate: not empty, reasonable length, not too many words
        if cleaned and len(cleaned.split()) <= 4 and 1 < len(cleaned) < 50:
            components.append(cleaned.lower())
    
    return components


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_batch(
    session: aiohttp.ClientSession,
    config: CategoryConfig,
    model: Literal["claude", "kimi"],
    existing: set[str],
    anthropic_key: str,
    openrouter_key: str,
    semaphore: asyncio.Semaphore,
) -> GenerationResult | None:
    """Generate a batch of components using specified model"""
    
    async with semaphore:
        try:
            category_context = make_category_context(config)
            user_message = make_user_message(existing)
            
            if model == "claude":
                response = await call_anthropic(
                    session, category_context, user_message, anthropic_key
                )
                components, token_info = parse_anthropic_response(response)
                
                return GenerationResult(
                    category=config.name,
                    model=model,
                    components=[c for c in components if c not in existing],
                    prompt_tokens=token_info["prompt_tokens"],
                    completion_tokens=token_info["completion_tokens"],
                    cached_tokens=token_info["cache_read_tokens"],
                    cache_write_tokens=token_info["cache_write_tokens"],
                    cache_read_tokens=token_info["cache_read_tokens"],
                    uncached_input_tokens=token_info["uncached_input_tokens"],
                )
            else:
                response = await call_openrouter_kimi(
                    session, category_context, user_message, openrouter_key
                )
                components, token_info = parse_openrouter_response(response)
                
                return GenerationResult(
                    category=config.name,
                    model=model,
                    components=[c for c in components if c not in existing],
                    prompt_tokens=token_info["prompt_tokens"],
                    completion_tokens=token_info["completion_tokens"],
                    cached_tokens=token_info.get("cached_tokens", 0),
                )
            
        except Exception as e:
            logger.error(f"Error generating {config.name} with {model}: {e}")
            return None


def save_checkpoint(state: GenerationState, checkpoint_path: Path) -> None:
    """Save intermediate generation state for resume capability"""
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "total_requests": state.total_requests,
            "total_tokens": state.total_tokens,
            "cached_tokens": state.total_cached_tokens,
            "errors": len(state.errors),
        },
        "components": {cat: sorted(list(words)) for cat, words in state.components.items()},
    }
    with open(checkpoint_path, "w") as f:
        yaml.dump(checkpoint, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path) -> GenerationState | None:
    """Load previous generation state from checkpoint"""
    if not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_path) as f:
            data = yaml.safe_load(f)
        
        state = GenerationState()
        state.components = {cat: set(words) for cat, words in data.get("components", {}).items()}
        state.total_requests = data.get("stats", {}).get("total_requests", 0)
        state.total_tokens = data.get("stats", {}).get("total_tokens", 0)
        state.total_cached_tokens = data.get("stats", {}).get("cached_tokens", 0)
        
        logger.info(f"Resumed from checkpoint: {checkpoint_path}")
        total = sum(len(c) for c in state.components.values())
        logger.info(f"  Loaded {total} existing components")
        
        return state
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


async def run_generation(
    categories: list[CategoryConfig],
    anthropic_key: str,
    openrouter_key: str,
    dry_run: bool = False,
    use_claude: bool = True,
    use_kimi: bool = True,
    checkpoint_path: Path | None = None,
    claude_cost_cap: float = DEFAULT_CLAUDE_COST_CAP,
) -> GenerationState:
    """
    Run the generation loop, alternating between models.
    
    Strategy:
    1. For each category, run 6 parallel requests with current model
    2. Collect unique results, add to pool
    3. Switch to other model, repeat
    4. Continue until target reached
    
    Supports resume from checkpoint if previous run was interrupted.
    """
    # Try to resume from checkpoint
    state = None
    if checkpoint_path:
        state = load_checkpoint(checkpoint_path)
    
    if state is None:
        state = GenerationState()
        # Initialize with seeds
        for config in categories:
            state.components[config.name] = set(config.seeds)
    else:
        # Ensure all requested categories exist in state
        for config in categories:
            if config.name not in state.components:
                state.components[config.name] = set(config.seeds)
    
    if dry_run:
        logger.info("=== DRY RUN - Showing prompts ===")
        for config in categories:
            category_context = make_category_context(config)
            user_message = make_user_message(state.components[config.name])
            logger.info(f"\n{'='*60}")
            logger.info(f"Category: {config.name}")
            logger.info(f"{'='*60}")
            logger.info(f"\nCached System Prompt ({len(SYSTEM_PROMPT_STATIC)} chars):")
            logger.info(f"{SYSTEM_PROMPT_STATIC[:300]}...")
            logger.info(f"\nCached Category Context ({len(category_context)} chars):")
            logger.info(f"{category_context[:500]}...")
            logger.info(f"\nDynamic User Message:")
            logger.info(user_message)
        return state
    
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    
    # Determine which models to use (mutable - Claude can be disabled mid-run)
    models: list[Literal["claude", "kimi"]] = []
    if use_claude:
        models.append("claude")
    if use_kimi:
        models.append("kimi")
    
    if not models:
        logger.error("No models enabled!")
        return state
    
    claude_disabled_by_cost = False  # Track if we hit the cost cap
    
    async with aiohttp.ClientSession() as session:
        iteration = 0
        max_iterations = 100  # Safety limit
        current_model_idx = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check if Claude should be disabled due to cost cap
            if not claude_disabled_by_cost and "claude" in models and state.claude_cost >= claude_cost_cap:
                logger.warning(f"\n⚠️  CLAUDE COST CAP REACHED: ${state.claude_cost:.2f} >= ${claude_cost_cap:.2f}")
                logger.warning("    Switching to Kimi-only mode for remaining generation")
                models.remove("claude")
                claude_disabled_by_cost = True
                current_model_idx = 0  # Reset to start of remaining models
                
                if not models:
                    logger.error("No models remaining after cost cap!")
                    break
            
            current_model = models[current_model_idx % len(models)]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration} - Model: {current_model.upper()}")
            logger.info(f"{'='*60}")
            
            # Check progress and find incomplete categories
            incomplete_categories = []
            for config in categories:
                current = len(state.components[config.name])
                if current < config.target_count:
                    incomplete_categories.append(config)
                    logger.info(f"  {config.name}: {current}/{config.target_count}")
                else:
                    logger.info(f"  {config.name}: {current}/{config.target_count} ✓")
            
            if not incomplete_categories:
                logger.info("\n🎉 All categories complete!")
                break
            
            # Generate batches for all incomplete categories with current model
            # 6 parallel requests per iteration
            tasks = []
            for config in incomplete_categories:
                tasks.append(
                    generate_batch(
                        session=session,
                        config=config,
                        model=current_model,
                        existing=state.components[config.name],
                        anthropic_key=anthropic_key,
                        openrouter_key=openrouter_key,
                        semaphore=semaphore,
                    )
                )
            
            # Run batch
            results = await asyncio.gather(*tasks)
            
            # Process results
            batch_added = 0
            batch_claude_cost = 0.0
            
            for result in results:
                if result is None:
                    state.errors.append(f"Failed {current_model}")
                    continue
                
                added = state.add_components(result.category, result.components)
                state.total_tokens += result.prompt_tokens + result.completion_tokens
                state.total_cached_tokens += result.cached_tokens
                state.total_requests += 1
                batch_added += added
                
                # Track Claude costs
                if result.model == "claude":
                    request_cost = state.add_claude_usage(
                        input_tokens=result.uncached_input_tokens,
                        cache_write_tokens=result.cache_write_tokens,
                        cache_read_tokens=result.cache_read_tokens,
                        output_tokens=result.completion_tokens,
                    )
                    batch_claude_cost += request_cost
                
                if added > 0:
                    cost_str = f", cost=${request_cost:.4f}" if result.model == "claude" else ""
                    cache_str = f"cache_read={result.cache_read_tokens}" if result.model == "claude" else f"cached={result.cached_tokens}"
                    logger.info(
                        f"  [{result.model}] {result.category}: +{added} new "
                        f"({cache_str}{cost_str})"
                    )
            
            logger.info(f"  Batch total: +{batch_added} new components")
            if batch_claude_cost > 0:
                logger.info(f"  Claude cost this batch: ${batch_claude_cost:.4f} (total: ${state.claude_cost:.2f}/{claude_cost_cap:.2f})")
            
            # Checkpoint every N iterations
            if checkpoint_path and iteration % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(state, checkpoint_path)
            
            # Alternate to next model
            current_model_idx += 1
            
            # Small delay
            await asyncio.sleep(REQUEST_DELAY_MS / 1000)
        
        # Final checkpoint
        if checkpoint_path:
            save_checkpoint(state, checkpoint_path)
    
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# FILE I/O
# ═══════════════════════════════════════════════════════════════════════════════


def load_categories(yaml_path: Path) -> list[CategoryConfig]:
    """Load category configurations from components_v2.yaml"""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    
    configs = []
    components = data.get("components", {})
    metadata = data.get("category_metadata", {})
    
    for cat_name, items in components.items():
        meta = metadata.get(cat_name, {})
        seeds = [item["word"] for item in items]
        
        configs.append(
            CategoryConfig(
                name=cat_name,
                description=meta.get("description", ""),
                generation_guidance=meta.get("generation_guidance", ""),
                anti_examples=meta.get("anti_examples", []),
                seeds=seeds,
                sub_clusters=meta.get("sub_clusters", []),
                target_count=meta.get("target_count", TARGET_PER_CATEGORY),
            )
        )
    
    return configs


def save_results(state: GenerationState, output_path: Path) -> None:
    """Save generated components to YAML file"""
    output = {
        "generated_at": datetime.now().isoformat(),
        "stats": {
            "total_requests": state.total_requests,
            "total_tokens": state.total_tokens,
            "cached_tokens": state.total_cached_tokens,
            "cache_hit_rate": (
                f"{state.total_cached_tokens / max(state.total_tokens, 1) * 100:.1f}%"
            ),
            "errors": len(state.errors),
        },
        "components": {},
    }
    
    for cat_name, components in state.components.items():
        output["components"][cat_name] = sorted(list(components))
    
    with open(output_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, width=120)
    
    logger.info(f"Saved results to {output_path}")


def merge_into_components_v2(state: GenerationState, yaml_path: Path) -> None:
    """Merge generated components back into components_v2.yaml"""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    
    for cat_name, new_components in state.components.items():
        if cat_name not in data["components"]:
            continue
        
        existing_words = {item["word"] for item in data["components"][cat_name]}
        
        for word in sorted(new_components):
            if word not in existing_words:
                data["components"][cat_name].append({
                    "word": word,
                    "opposite": None,
                })
                existing_words.add(word)
    
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, width=120)
    
    logger.info(f"Merged results into {yaml_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate components for prompt categories using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available categories and exit",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Specific categories to generate (default: all)",
    )
    parser.add_argument(
        "--model",
        choices=["claude", "kimi", "both"],
        default="both",
        help="Model(s) to use for generation",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_PER_CATEGORY,
        help=f"Target components per category (default: {TARGET_PER_CATEGORY})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show prompts without making API calls",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for generated components",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge results directly into components_v2.yaml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--claude-cost-cap",
        type=float,
        default=DEFAULT_CLAUDE_COST_CAP,
        help=f"Max USD to spend on Claude before switching to Kimi-only (default: ${DEFAULT_CLAUDE_COST_CAP:.2f})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load categories
    project_root = Path(__file__).parent.parent.parent
    yaml_path = project_root / "prompts" / "components_v2.yaml"
    
    if not yaml_path.exists():
        logger.error(f"Components file not found: {yaml_path}")
        sys.exit(1)
    
    # Handle --list
    if args.list:
        categories = load_categories(yaml_path)
        print("\nAvailable categories:")
        print("-" * 50)
        for cat in categories:
            print(f"  {cat.name}")
            print(f"    Seeds: {len(cat.seeds)} | Target: {cat.target_count}")
            if cat.description:
                print(f"    {cat.description[:70]}...")
            print()
        sys.exit(0)
    
    # Check API keys
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    
    use_claude = args.model in ("claude", "both")
    use_kimi = args.model in ("kimi", "both")
    
    if not args.dry_run:
        if use_claude and not anthropic_key:
            logger.error("ANTHROPIC_API_KEY environment variable not set")
            logger.error("Get your key at: https://console.anthropic.com/")
            sys.exit(1)
        if use_kimi and not openrouter_key:
            logger.error("OPENROUTER_API_KEY environment variable not set")
            logger.error("Get your key at: https://openrouter.ai/keys")
            sys.exit(1)
    
    # Load and filter categories
    categories = load_categories(yaml_path)
    
    if args.categories:
        categories = [c for c in categories if c.name in args.categories]
        if not categories:
            logger.error(f"No matching categories found: {args.categories}")
            sys.exit(1)
    
    # Override target count
    for cat in categories:
        cat.target_count = args.target
    
    # Determine checkpoint path
    checkpoint_path = project_root / "prompts" / "generation_checkpoint.yaml" if args.resume else None
    
    logger.info("Starting component generation")
    logger.info(f"  Categories: {[c.name for c in categories]}")
    logger.info(f"  Models: claude={use_claude}, kimi={use_kimi}")
    logger.info(f"  Target per category: {args.target}")
    logger.info(f"  Parallel requests: {MAX_PARALLEL_REQUESTS}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    if use_claude:
        logger.info(f"  Claude cost cap: ${args.claude_cost_cap:.2f}")
    if checkpoint_path:
        logger.info(f"  Checkpoint: {checkpoint_path}")
    
    # Run generation
    state = asyncio.run(
        run_generation(
            categories=categories,
            anthropic_key=anthropic_key,
            openrouter_key=openrouter_key,
            dry_run=args.dry_run,
            use_claude=use_claude,
            use_kimi=use_kimi,
            checkpoint_path=checkpoint_path,
            claude_cost_cap=args.claude_cost_cap,
        )
    )
    
    if args.dry_run:
        return
    
    # Report results
    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    
    total_generated = sum(len(c) for c in state.components.values())
    logger.info(f"Total unique components: {total_generated}")
    logger.info(f"API requests: {state.total_requests}")
    logger.info(f"Total tokens: {state.total_tokens:,}")
    logger.info(f"Cached tokens: {state.total_cached_tokens:,}")
    if state.total_tokens > 0:
        cache_rate = state.total_cached_tokens / state.total_tokens * 100
        logger.info(f"Cache hit rate: {cache_rate:.1f}%")
    
    # Claude cost breakdown
    if state.claude_cost > 0:
        logger.info(f"\nClaude Cost Breakdown:")
        logger.info(f"  Input tokens (uncached): {state.claude_input_tokens:,} (${state.claude_input_tokens * CLAUDE_PRICING['input'] / 1_000_000:.4f})")
        logger.info(f"  Cache write tokens:      {state.claude_cache_write_tokens:,} (${state.claude_cache_write_tokens * CLAUDE_PRICING['cache_write'] / 1_000_000:.4f})")
        logger.info(f"  Cache read tokens:       {state.claude_cache_read_tokens:,} (${state.claude_cache_read_tokens * CLAUDE_PRICING['cache_read'] / 1_000_000:.4f})")
        logger.info(f"  Output tokens:           {state.claude_output_tokens:,} (${state.claude_output_tokens * CLAUDE_PRICING['output'] / 1_000_000:.4f})")
        logger.info(f"  TOTAL CLAUDE COST:       ${state.claude_cost:.2f}")
    
    if state.errors:
        logger.warning(f"Errors: {len(state.errors)}")
    
    for cat_name, components in state.components.items():
        logger.info(f"  {cat_name}: {len(components)}")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_root / "prompts" / f"generated_components_{timestamp}.yaml"
    
    save_results(state, output_path)
    
    # Optionally merge
    if args.merge:
        merge_into_components_v2(state, yaml_path)


if __name__ == "__main__":
    main()
