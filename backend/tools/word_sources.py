"""
Word Sources for Component Diversity Mining

Large curated word pools for each prompt category, sourced from:
- Linguistic databases (descriptive terms)
- Art/photography terminology
- Visual concept vocabularies

These are pre-filtered for terms likely to be meaningful to CLIP/Stable Diffusion.
Abstract or overly technical terms are excluded.

Combinatorics note:
    With N components per category and T templates using ~5 categories each:
    Total combinations ≈ T × N^5
    
    - 12 components: 12 × 12^5 = 3 million combinations (~6 years at 1/min)
    - 25 components: 12 × 25^5 = 117 million combinations (~222 years)
    - 50 components: 12 × 50^5 = 3.75 billion combinations (~7,000 years)
    
    We recommend 20-40 curated components per category for the sweet spot of
    diversity, quality, and manageability.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# ADJECTIVES - Visual/descriptive qualities
# Sources: Curated from descriptive writing, visual arts terminology
# ═══════════════════════════════════════════════════════════════════════════════
ADJECTIVES = [
    # Original
    "ethereal", "digital", "fractured", "luminous", "corrupted",
    "geometric", "translucent", "ancient", "mechanical", "spectral",
    "crystalline", "hollow",
    
    # Temperature/State
    "molten", "frozen", "burning", "smoldering", "glacial", "volcanic",
    "liquid", "gaseous", "solid", "vaporized", "crystallized", "petrified",
    
    # Texture/Surface
    "smooth", "rough", "jagged", "polished", "corroded", "weathered",
    "glossy", "matte", "textured", "porous", "scaled", "feathered",
    "woven", "braided", "knotted", "tangled", "twisted", "coiled",
    
    # Light/Opacity
    "opaque", "transparent", "iridescent", "reflective", "absorbing",
    "glowing", "dim", "radiant", "shadowed", "backlit", "silhouetted",
    "phosphorescent", "fluorescent", "bioluminescent", "incandescent",
    
    # Age/Decay
    "ancient", "primordial", "fossilized", "decayed", "rotting", "withered",
    "pristine", "worn", "eroded", "crumbling", "rusted", "tarnished",
    "oxidized", "calcified", "minerite", "petrified",
    
    # Structure
    "skeletal", "hollow", "solid", "dense", "sparse", "layered",
    "fragmented", "shattered", "splintered", "compressed", "expanded",
    "angular", "curved", "spiral", "branching", "symmetrical", "asymmetrical",
    
    # Organic/Inorganic
    "organic", "synthetic", "artificial", "natural", "biomechanical",
    "cybernetic", "robotic", "fleshy", "metallic", "wooden", "stone",
    
    # Movement/Energy
    "static", "flowing", "pulsing", "vibrating", "oscillating", "rotating",
    "floating", "suspended", "falling", "rising", "expanding", "contracting",
    
    # Scale
    "microscopic", "miniature", "colossal", "monumental", "infinite",
    "compressed", "stretched", "distorted", "warped", "bent",
    
    # Emotional quality
    "menacing", "serene", "chaotic", "orderly", "violent", "peaceful",
    "mysterious", "familiar", "alien", "uncanny", "dreamlike", "nightmarish",
    
    # Technical/Digital
    "pixelated", "vectorized", "procedural", "generative", "algorithmic",
    "glitched", "corrupted", "fragmented", "interpolated", "extruded",
]

# ═══════════════════════════════════════════════════════════════════════════════
# NOUNS - Visual subjects (concrete, depictable)
# Filtered for things that render well in image generation
# ═══════════════════════════════════════════════════════════════════════════════
NOUNS = [
    # Original
    "angel", "figure", "skull", "cathedral", "serpent",
    "mask", "tower", "eye", "hand", "tree",
    
    # Anatomy
    "skeleton", "spine", "ribcage", "pelvis", "vertebra",
    "skull", "jaw", "teeth", "bone", "marrow",
    "heart", "lungs", "brain", "veins", "arteries",
    "skin", "muscle", "tendon", "nerve", "cell",
    
    # Body parts
    "hand", "fingers", "palm", "fist", "arm",
    "face", "eye", "mouth", "lips", "tongue",
    "head", "neck", "torso", "limb", "foot",
    
    # Creatures
    "angel", "demon", "ghost", "spirit", "phantom",
    "dragon", "serpent", "wolf", "crow", "moth",
    "whale", "jellyfish", "octopus", "spider", "beetle",
    "butterfly", "bird", "fish", "worm", "larva",
    
    # Architecture
    "cathedral", "temple", "tower", "bridge", "arch",
    "column", "pillar", "staircase", "doorway", "window",
    "vault", "dome", "spire", "buttress", "portal",
    "corridor", "chamber", "hall", "throne", "altar",
    
    # Nature
    "tree", "root", "branch", "leaf", "flower",
    "mountain", "cliff", "canyon", "cave", "crater",
    "ocean", "wave", "river", "waterfall", "pool",
    "cloud", "storm", "lightning", "aurora", "sun", "moon",
    
    # Objects
    "mirror", "clock", "key", "door", "gate",
    "chain", "rope", "thread", "web", "net",
    "crystal", "gem", "prism", "orb", "sphere",
    "blade", "sword", "needle", "thorn", "spike",
    
    # Mechanical
    "machine", "engine", "gear", "piston", "wheel",
    "circuit", "wire", "cable", "antenna", "satellite",
    "robot", "automaton", "drone", "mech", "cyborg",
    
    # Abstract shapes
    "spiral", "helix", "vortex", "fractal", "mandala",
    "grid", "lattice", "mesh", "matrix", "pattern",
    "ring", "loop", "knot", "tangle", "weave",
    
    # Celestial
    "star", "planet", "nebula", "galaxy", "comet",
    "asteroid", "meteor", "eclipse", "constellation", "void",
    
    # Symbols
    "halo", "crown", "wings", "horn", "tail",
    "cross", "circle", "triangle", "pentagram", "sigil",
]

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS - Visual color descriptions
# Sources: XKCD color survey, art terminology, material colors
# ═══════════════════════════════════════════════════════════════════════════════
COLORS = [
    # Original
    "monochrome", "cyan accents", "deep crimson", "gold and black",
    "pale violet", "oxidized copper", "ash white", "bioluminescent",
    "rust and bone", "infrared",
    
    # Monochromatic
    "pure white", "bone white", "ivory", "cream", "off-white",
    "silver gray", "charcoal", "slate", "obsidian", "pitch black",
    "void black", "matte black", "glossy black",
    
    # Reds
    "blood red", "crimson", "scarlet", "vermillion", "rust",
    "burgundy", "maroon", "wine", "cherry", "rose",
    "coral", "salmon", "brick red", "terracotta",
    
    # Oranges
    "burnt orange", "amber", "copper", "bronze", "gold",
    "tangerine", "apricot", "peach", "rust orange", "flame",
    
    # Yellows
    "golden yellow", "mustard", "ochre", "canary", "lemon",
    "sulfur", "honey", "brass", "champagne", "blonde",
    
    # Greens
    "emerald", "jade", "olive", "forest green", "moss",
    "sage", "mint", "lime", "chartreuse", "teal",
    "verdigris", "patina", "malachite", "toxic green", "neon green",
    
    # Blues
    "navy", "cobalt", "sapphire", "azure", "cerulean",
    "cyan", "turquoise", "teal", "aquamarine", "ice blue",
    "midnight blue", "royal blue", "electric blue", "steel blue",
    
    # Purples
    "violet", "lavender", "lilac", "plum", "grape",
    "amethyst", "mauve", "magenta", "fuchsia", "orchid",
    "royal purple", "deep purple", "ultraviolet",
    
    # Pinks
    "hot pink", "neon pink", "blush", "rose", "coral pink",
    "dusty pink", "salmon pink", "bubblegum",
    
    # Browns
    "chocolate", "coffee", "espresso", "mahogany", "walnut",
    "caramel", "tan", "khaki", "beige", "sand",
    "umber", "sienna", "sepia",
    
    # Metallics
    "chrome", "silver", "gold", "bronze", "copper",
    "brass", "platinum", "iron", "steel", "mercury",
    "holographic", "iridescent", "opalescent", "pearlescent",
    
    # Special effects
    "bioluminescent", "phosphorescent", "fluorescent", "neon",
    "infrared", "ultraviolet", "thermal", "radioactive",
    "prismatic", "rainbow", "gradient", "duochrome",
    
    # Combinations
    "black and white", "black and gold", "red and black",
    "blue and orange", "purple and gold", "green and pink",
    "fire and ice", "rust and bone", "blood and gold",
]

# ═══════════════════════════════════════════════════════════════════════════════
# STYLES - Art styles, techniques, aesthetics
# Sources: Art history, photography, digital art, printmaking
# ═══════════════════════════════════════════════════════════════════════════════
STYLES = [
    # Original
    "wireframe", "ink wash", "glitch art", "technical diagram",
    "double exposure", "datamosh", "photogram", "xray aesthetic",
    "scan lines", "architectural render",
    
    # Traditional fine art
    "oil painting", "watercolor", "acrylic", "gouache", "tempera",
    "fresco", "encaustic", "impasto", "glazing", "sfumato",
    
    # Drawing
    "pencil sketch", "charcoal", "graphite", "conte crayon",
    "ink drawing", "pen and ink", "crosshatching", "stippling",
    
    # Printmaking
    "woodcut", "linocut", "etching", "engraving", "lithograph",
    "screen print", "risograph", "monoprint", "collagraph",
    "cyanotype", "photogram", "blueprint",
    
    # Photography
    "daguerreotype", "tintype", "albumen print", "platinum print",
    "polaroid", "lomography", "infrared photography", "long exposure",
    "macro photography", "aerial photography", "underwater photography",
    
    # Digital/Technical
    "3D render", "ray tracing", "path tracing", "ambient occlusion",
    "wireframe", "polygon mesh", "voxel art", "isometric",
    "vector art", "pixel art", "ASCII art", "text art",
    
    # Effects/Filters
    "glitch art", "datamosh", "compression artifacts", "VHS aesthetic",
    "scan lines", "chromatic aberration", "film grain", "noise",
    "posterization", "solarization", "halftone", "duotone",
    
    # Movements/Schools
    "baroque", "renaissance", "impressionist", "expressionist",
    "surrealist", "cubist", "abstract", "minimalist",
    "art nouveau", "art deco", "brutalist", "gothic",
    "cyberpunk", "steampunk", "dieselpunk", "biopunk",
    
    # Scientific/Medical
    "xray", "CT scan", "MRI", "ultrasound", "thermal imaging",
    "microscopy", "electron microscope", "spectroscopy",
    "topographic map", "satellite imagery", "sonar",
    
    # Illustration
    "comic book", "manga", "graphic novel", "storyboard",
    "concept art", "matte painting", "book illustration",
    "scientific illustration", "botanical illustration", "anatomical drawing",
]

# ═══════════════════════════════════════════════════════════════════════════════
# MOODS - Emotional atmospheres
# Sources: Emotion lexicons, film terminology, music descriptors
# ═══════════════════════════════════════════════════════════════════════════════
MOODS = [
    # Original
    "melancholic", "ominous", "serene", "haunting", "transcendent",
    "desolate", "liminal", "fevered", "sacred", "clinical",
    
    # Positive/Uplifting
    "euphoric", "blissful", "ecstatic", "triumphant", "majestic",
    "serene", "peaceful", "tranquil", "harmonious", "sublime",
    "hopeful", "uplifting", "liberating", "exhilarating",
    
    # Negative/Dark
    "ominous", "dreadful", "terrifying", "horrific", "nightmarish",
    "oppressive", "suffocating", "crushing", "devastating", "apocalyptic",
    "mournful", "grieving", "sorrowful", "tragic", "despairing",
    
    # Mysterious/Liminal
    "mysterious", "enigmatic", "cryptic", "arcane", "occult",
    "liminal", "transitional", "threshold", "between worlds",
    "uncanny", "surreal", "dreamlike", "hypnagogic", "hallucinatory",
    
    # Energy/Intensity
    "intense", "explosive", "violent", "aggressive", "brutal",
    "chaotic", "frantic", "manic", "feverish", "delirious",
    "calm", "still", "quiet", "hushed", "silent",
    
    # Atmosphere
    "eerie", "ghostly", "spectral", "haunted", "cursed",
    "sacred", "holy", "divine", "celestial", "ethereal",
    "profane", "corrupt", "decadent", "sinister", "malevolent",
    
    # Temporal
    "ancient", "primordial", "timeless", "eternal", "ephemeral",
    "nostalgic", "melancholic", "wistful", "bittersweet",
    "futuristic", "dystopian", "utopian", "post-apocalyptic",
    
    # Spatial
    "vast", "infinite", "boundless", "claustrophobic", "confined",
    "isolated", "abandoned", "forgotten", "hidden", "secret",
    "exposed", "vulnerable", "protected", "sheltered",
    
    # Medical/Clinical
    "clinical", "sterile", "antiseptic", "surgical", "diagnostic",
    "dissected", "examined", "preserved", "catalogued",
]

# ═══════════════════════════════════════════════════════════════════════════════
# EFFECTS - Visual effects and transformations
# Sources: VFX terminology, photography, digital art, film
# ═══════════════════════════════════════════════════════════════════════════════
EFFECTS = [
    # Original
    "particle dissolution", "pixel scatter", "chromatic aberration",
    "smoke tendrils", "light leak", "data corruption", "refraction",
    "echo trails", "interference patterns", "scan distortion",
    
    # Dissolution/Disintegration
    "particle dissolution", "ash scatter", "sand dispersal",
    "pixel scatter", "voxel explosion", "fragment burst",
    "vapor trail", "smoke wisps", "mist fade",
    
    # Distortion
    "ripple distortion", "wave distortion", "heat shimmer",
    "lens distortion", "barrel distortion", "pincushion",
    "stretch", "squeeze", "twist", "spiral warp",
    "glitch displacement", "databend", "pixel sort",
    
    # Light effects
    "lens flare", "light leak", "god rays", "volumetric light",
    "bloom", "glow", "halo", "corona", "caustics",
    "refraction", "reflection", "specular highlight",
    "bokeh", "depth blur", "tilt shift",
    
    # Film/Video
    "film grain", "film scratch", "dust particles",
    "VHS tracking", "analog noise", "static",
    "interlacing", "scan lines", "rolling shutter",
    "motion blur", "frame drag", "ghosting",
    
    # Digital artifacts
    "compression artifacts", "JPEG artifacts", "macroblocking",
    "color banding", "posterization", "dithering",
    "aliasing", "moiré pattern", "interference",
    
    # Color effects
    "chromatic aberration", "color fringing", "color shift",
    "color separation", "RGB split", "channel offset",
    "duotone", "tritone", "color grading",
    "desaturation", "oversaturation", "color inversion",
    
    # Overlay/Composite
    "double exposure", "multiple exposure", "overlay blend",
    "screen blend", "multiply blend", "difference blend",
    "texture overlay", "pattern overlay", "noise overlay",
    
    # Organic
    "smoke tendrils", "fire trails", "water ripples",
    "ice crystals", "frost patterns", "lightning bolts",
    "root growth", "vein spread", "crack propagation",
    
    # Geometric
    "grid overlay", "wireframe", "contour lines",
    "voronoi", "delaunay", "fractal growth",
    "recursive pattern", "tessellation", "kaleidoscope",
]

# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS - Environments and locations
# Sources: Architecture, geography, film locations, game environments
# ═══════════════════════════════════════════════════════════════════════════════
SETTINGS = [
    # Original
    "void", "cathedral interior", "data center", "ruined temple",
    "liminal corridor", "frozen lake", "server room", "ossuary",
    "threshold", "horizon line",
    
    # Religious/Sacred
    "cathedral interior", "gothic church", "monastery", "cloister",
    "temple", "shrine", "sanctuary", "chapel", "crypt",
    "altar room", "bell tower", "sacristy", "nave",
    
    # Ruins/Decay
    "ruined temple", "collapsed cathedral", "crumbling palace",
    "overgrown ruins", "abandoned factory", "derelict ship",
    "forgotten bunker", "decayed mansion", "rotting greenhouse",
    
    # Industrial
    "factory floor", "warehouse", "power plant", "refinery",
    "foundry", "steel mill", "assembly line", "boiler room",
    "machine room", "control room", "maintenance tunnel",
    
    # Medical/Scientific
    "operating theater", "autopsy room", "morgue", "laboratory",
    "specimen room", "clean room", "quarantine chamber",
    "cryogenic facility", "research station",
    
    # Digital/Tech
    "server room", "data center", "server farm", "mainframe",
    "network hub", "control center", "mission control",
    "virtual space", "cyberspace", "digital void",
    
    # Natural
    "deep forest", "ancient grove", "dead forest", "petrified forest",
    "mountain peak", "volcanic crater", "glacier", "ice cave",
    "underwater cave", "coral reef", "deep ocean", "abyssal plain",
    "desert dunes", "salt flat", "canyon", "ravine",
    
    # Liminal/Transitional
    "empty corridor", "infinite hallway", "stairwell", "elevator shaft",
    "threshold", "doorway", "archway", "passage",
    "waiting room", "empty lobby", "abandoned mall",
    "empty parking garage", "subway platform", "train station",
    
    # Cosmic
    "void", "nebula", "asteroid field", "planetary surface",
    "orbital station", "space station", "airlock",
    "event horizon", "singularity", "cosmic web",
    
    # Surreal/Abstract
    "infinite plane", "endless grid", "mirror dimension",
    "fractal landscape", "impossible architecture",
    "tesseract interior", "non-euclidean space",
    
    # Domestic (unsettling)
    "empty bedroom", "flooded basement", "attic space",
    "dark staircase", "long hallway", "abandoned nursery",
    "old bathroom", "forgotten kitchen",
    
    # Containment
    "prison cell", "padded room", "isolation chamber",
    "specimen jar", "display case", "terrarium",
    "aquarium", "vivarium", "observation deck",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Combined lookup
# ═══════════════════════════════════════════════════════════════════════════════
WORD_POOLS = {
    "adjective": ADJECTIVES,
    "noun": NOUNS,
    "color": COLORS,
    "style": STYLES,
    "mood": MOODS,
    "effect": EFFECTS,
    "setting": SETTINGS,
}

def get_word_pool(category: str) -> list[str]:
    """Get word pool for a category, with deduplication"""
    words = WORD_POOLS.get(category, [])
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for word in words:
        word_lower = word.lower()
        if word_lower not in seen:
            seen.add(word_lower)
            unique.append(word)
    return unique


def get_pool_stats() -> dict[str, int]:
    """Get counts for each word pool"""
    return {category: len(get_word_pool(category)) for category in WORD_POOLS}


# ═══════════════════════════════════════════════════════════════════════════════
# Part-of-Speech Filtering
# ═══════════════════════════════════════════════════════════════════════════════

def get_wordnet_pos(word: str) -> set[str]:
    """
    Get WordNet parts of speech for a word
    
    Returns set of: 'n' (noun), 'v' (verb), 'a' (adjective), 'r' (adverb), 's' (adjective satellite)
    Empty set if word not found or NLTK not available.
    """
    try:
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets(word)
        return {s.pos() for s in synsets}
    except Exception:
        return set()


def is_adjective(word: str) -> bool:
    """Check if word can be used as an adjective"""
    pos = get_wordnet_pos(word)
    # 'a' = adjective, 's' = adjective satellite (similar meaning)
    return bool(pos & {'a', 's'})


def is_noun(word: str) -> bool:
    """Check if word can be used as a noun"""
    pos = get_wordnet_pos(word)
    return 'n' in pos


def filter_by_pos(words: list[str], target_pos: str) -> list[str]:
    """
    Filter words by part of speech
    
    Args:
        words: List of words to filter
        target_pos: Target POS - 'adjective', 'noun', 'verb', 'adverb'
        
    Returns:
        Words that can be used as the target POS
    """
    pos_check = {
        'adjective': is_adjective,
        'noun': is_noun,
        'verb': lambda w: 'v' in get_wordnet_pos(w),
        'adverb': lambda w: 'r' in get_wordnet_pos(w),
    }
    
    checker = pos_check.get(target_pos)
    if not checker:
        return words  # No filter for unknown POS
    
    return [w for w in words if checker(w)]


def ensure_nltk_data():
    """Download NLTK WordNet data if not present"""
    try:
        from nltk.corpus import wordnet
        # Test if data is available
        wordnet.synsets('test')
    except LookupError:
        import nltk
        print("Downloading NLTK WordNet data...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)  # Open Multilingual WordNet
    except ImportError:
        print("NLTK not installed. Run: pip install nltk")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Online Word Sources
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_from_datamuse(
    query_type: str,
    seed_word: str,
    max_results: int = 100
) -> list[str]:
    """
    Fetch related words from Datamuse API (free, no key required)
    
    Query types:
        ml  - "means like" (semantic similarity)
        rel_syn - synonyms
        rel_trg - "triggers" (associated words)
        rel_jja - adjectives that describe the noun
        rel_jjb - nouns described by the adjective
        
    Example:
        fetch_from_datamuse("ml", "ethereal") -> words meaning like ethereal
        fetch_from_datamuse("rel_jja", "sky") -> adjectives for sky
    """
    import urllib.request
    import json
    
    url = f"https://api.datamuse.com/words?{query_type}={seed_word}&max={max_results}"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            return [item["word"] for item in data if " " not in item["word"]]
    except Exception as e:
        print(f"Datamuse fetch failed: {e}")
        return []


def fetch_adjectives_online(seed_words: list[str] | None = None, max_per_seed: int = 50) -> list[str]:
    """Fetch adjectives from Datamuse using seed words"""
    if seed_words is None:
        seed_words = ["ethereal", "ancient", "corrupted", "luminous", "mechanical", 
                      "dark", "bright", "organic", "digital", "frozen"]
    
    all_words = set()
    for seed in seed_words:
        words = fetch_from_datamuse("ml", seed, max_per_seed)
        all_words.update(words)
        # Also get adjectives that trigger from this concept
        words = fetch_from_datamuse("rel_trg", seed, max_per_seed)
        all_words.update(words)
    
    return list(all_words)


def fetch_nouns_online(seed_words: list[str] | None = None, max_per_seed: int = 50) -> list[str]:
    """Fetch concrete nouns from Datamuse"""
    if seed_words is None:
        seed_words = ["skull", "tower", "angel", "serpent", "crystal",
                      "machine", "tree", "flame", "shadow", "mirror"]
    
    all_words = set()
    for seed in seed_words:
        words = fetch_from_datamuse("ml", seed, max_per_seed)
        all_words.update(words)
    
    return list(all_words)


def fetch_moods_online(seed_words: list[str] | None = None, max_per_seed: int = 50) -> list[str]:
    """Fetch mood/emotion words from Datamuse"""
    if seed_words is None:
        seed_words = ["melancholic", "ominous", "serene", "haunting", "euphoric",
                      "dread", "peace", "chaos", "sacred", "violent"]
    
    all_words = set()
    for seed in seed_words:
        words = fetch_from_datamuse("ml", seed, max_per_seed)
        all_words.update(words)
    
    return list(all_words)


def expand_pool_online(
    category: str, 
    max_words: int = 200,
    filter_pos: bool = True
) -> list[str]:
    """
    Expand a word pool using online sources
    
    Combines existing curated pool with fetched words.
    Optionally filters by part-of-speech to remove wrong word types.
    
    Args:
        category: Category to expand
        max_words: Maximum words to return
        filter_pos: If True, filter by appropriate POS (requires NLTK)
        
    Returns:
        Deduplicated list of candidate words
    """
    existing = get_word_pool(category)
    
    # Fetch based on category
    fetched = []
    target_pos = None  # POS to filter to
    
    if category == "adjective":
        fetched = fetch_adjectives_online(seed_words=existing[:10])
        target_pos = "adjective"
    elif category == "noun":
        fetched = fetch_nouns_online(seed_words=existing[:10])
        target_pos = "noun"
    elif category == "mood":
        fetched = fetch_moods_online(seed_words=existing[:10])
        target_pos = "adjective"  # Moods should be adjectives
    # Note: color, style, effect, setting don't have good online sources
    # They need domain-specific curation
    
    # Combine and deduplicate
    combined = existing + fetched
    seen = set()
    unique = []
    for word in combined:
        word_lower = word.lower()
        if word_lower not in seen and len(word) > 2:
            seen.add(word_lower)
            unique.append(word)
    
    # Apply POS filter if requested
    if filter_pos and target_pos and fetched:
        if ensure_nltk_data():
            before_count = len(unique)
            unique = filter_by_pos(unique, target_pos)
            filtered_count = before_count - len(unique)
            if filtered_count > 0:
                print(f"  POS filter removed {filtered_count} non-{target_pos}s")
    
    return unique[:max_words]


# ═══════════════════════════════════════════════════════════════════════════════
# Combinatorics Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_combinations(
    components_per_category: int | dict[str, int],
    num_templates: int = 12,
    categories_per_template: int = 5
) -> dict[str, any]:
    """
    Estimate total unique prompt combinations
    
    Args:
        components_per_category: Either a single int or dict mapping category to count
        num_templates: Number of templates
        categories_per_template: Average categories used per template
        
    Returns:
        Dict with combination counts and time estimates
    """
    if isinstance(components_per_category, int):
        n = components_per_category
    else:
        # Use geometric mean for mixed sizes
        import math
        counts = list(components_per_category.values())
        n = int(math.exp(sum(math.log(c) for c in counts) / len(counts)))
    
    combinations_per_template = n ** categories_per_template
    total_combinations = num_templates * combinations_per_template
    
    # Time estimates at 1 generation per minute
    minutes_per_year = 525600
    years_to_exhaust = total_combinations / minutes_per_year
    
    return {
        "components_per_category": n,
        "combinations_per_template": combinations_per_template,
        "total_combinations": total_combinations,
        "years_to_exhaust_at_1_per_min": years_to_exhaust,
        "human_readable": format_large_number(total_combinations),
    }


def format_large_number(n: int) -> str:
    """Format large number for human readability"""
    if n >= 1e12:
        return f"{n/1e12:.1f} trillion"
    elif n >= 1e9:
        return f"{n/1e9:.1f} billion"
    elif n >= 1e6:
        return f"{n/1e6:.1f} million"
    elif n >= 1e3:
        return f"{n/1e3:.1f} thousand"
    else:
        return str(n)


def recommend_pool_size(
    target_years: float = 100,
    num_templates: int = 12,
    categories_per_template: int = 5,
    generations_per_minute: float = 1.0
) -> int:
    """
    Calculate recommended components per category to last N years
    
    Args:
        target_years: How many years before theoretical repetition
        num_templates: Number of templates
        categories_per_template: Average categories per template
        generations_per_minute: Generation rate
        
    Returns:
        Recommended components per category
    """
    import math
    
    generations_needed = target_years * 525600 * generations_per_minute
    combinations_per_template_needed = generations_needed / num_templates
    n = combinations_per_template_needed ** (1 / categories_per_template)
    
    return max(10, int(math.ceil(n)))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Word pool statistics and tools")
    parser.add_argument("--fetch", metavar="CATEGORY", help="Fetch words online for category")
    parser.add_argument("--estimate", type=int, metavar="N", help="Estimate combinations with N components")
    parser.add_argument("--recommend", type=float, metavar="YEARS", help="Recommend pool size for N years")
    args = parser.parse_args()
    
    if args.fetch:
        print(f"Fetching words for '{args.fetch}'...")
        words = expand_pool_online(args.fetch)
        print(f"Found {len(words)} words:")
        for word in sorted(words)[:50]:
            print(f"  - {word}")
        if len(words) > 50:
            print(f"  ... and {len(words) - 50} more")
    
    elif args.estimate:
        stats = estimate_combinations(args.estimate)
        print(f"With {args.estimate} components per category:")
        print(f"  Total combinations: {stats['human_readable']}")
        print(f"  Years to exhaust (1/min): {stats['years_to_exhaust_at_1_per_min']:.1f}")
    
    elif args.recommend:
        n = recommend_pool_size(target_years=args.recommend)
        print(f"To last {args.recommend} years at 1 gen/min:")
        print(f"  Need ~{n} components per category")
        stats = estimate_combinations(n)
        print(f"  This gives {stats['human_readable']} combinations")
    
    else:
        # Default: print statistics
        print("Word Pool Statistics:")
        print("-" * 30)
        total = 0
        for category, count in sorted(get_pool_stats().items()):
            print(f"  {category:<12}: {count:>4} words")
            total += count
        print("-" * 30)
        print(f"  {'TOTAL':<12}: {total:>4} words")
        
        print("\nCombination Estimates:")
        for n in [12, 25, 50, 100]:
            stats = estimate_combinations(n)
            print(f"  {n:>3} components: {stats['human_readable']:>15} ({stats['years_to_exhaust_at_1_per_min']:.0f} years)")

