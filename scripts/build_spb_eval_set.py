#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build SPB evaluation task set from PersonaExt data.

Pipeline:
1. Download PersonaExt raw data → tests/fixtures/spb_eval/raw/
2. Apply mapping.json to convert triplets → SPB schema
3. Score & rank by selection criteria
4. Output top 20 personas to tests/fixtures/spb_eval/personas/pe_XXX.json

Usage:
    python scripts/build_spb_eval_set.py [--skip-download] [--count 20]
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "spb_eval"
RAW_DIR = FIXTURES_DIR / "raw"
PERSONAS_DIR = FIXTURES_DIR / "personas"
MAPPING_PATH = FIXTURES_DIR / "mapping.json"

# PersonaExt triplet → SPB field mapping (simplified for synthetic data)
_PREDICATE_MAP = {
    "have_profession": ("background", "occupation", "str"),
    "have_hobby": ("interests", "hobbies", "list"),
    "like_to_do": ("interests", "hobbies", "list"),
    "like": ("interests", "topics", "list"),
    "favorite_color": ("interests", "topics", "list"),
    "favorite_animal": ("interests", "topics", "list"),
    "favorite_food": ("interests", "topics", "list"),
    "favorite_music": ("interests", "topics", "list"),
    "favorite_movie": ("interests", "topics", "list"),
    "favorite_book": ("interests", "topics", "list"),
    "favorite_sport": ("interests", "topics", "list"),
    "have_tool": ("preferences", "tools", "list"),
    "own_device": ("preferences", "tools", "list"),
}


def _generate_synthetic_personas(count: int = 20) -> list[dict]:
    """Generate synthetic persona data for SPB evaluation.

    When PersonaExt raw data is unavailable, generates diverse synthetic
    personas that meet the selection criteria. Each persona has:
    - Ground truth profile fields
    - 5-turn dialogue where profile info is naturally revealed
    """
    random.seed(42)

    names = [
        "Alice", "Bob", "Carol", "David", "Eve",
        "Frank", "Grace", "Henry", "Iris", "Jack",
        "Karen", "Leo", "Maya", "Nick", "Olivia",
        "Paul", "Quinn", "Rachel", "Sam", "Tara",
    ]
    occupations = [
        "frontend engineer", "data scientist", "product manager",
        "UX designer", "DevOps engineer", "machine learning researcher",
        "mobile developer", "security analyst", "technical writer",
        "game developer", "database administrator", "cloud architect",
        "QA engineer", "embedded systems engineer", "bioinformatics researcher",
        "full-stack developer", "systems architect", "AI ethicist",
        "robotics engineer", "blockchain developer",
    ]
    hobbies_pool = [
        ["hiking", "photography"], ["reading", "cooking"],
        ["gaming", "streaming"], ["painting", "yoga"],
        ["rock climbing", "cycling"], ["gardening", "baking"],
        ["chess", "podcasting"], ["surfing", "diving"],
        ["knitting", "writing"], ["running", "meditation"],
    ]
    topics_pool = [
        ["machine learning", "Python"], ["web development", "TypeScript"],
        ["cybersecurity", "privacy"], ["open source", "Linux"],
        ["AI ethics", "philosophy"], ["cloud computing", "Kubernetes"],
        ["data visualization", "R"], ["mobile apps", "Flutter"],
        ["distributed systems", "Rust"], ["NLP", "transformers"],
    ]
    tools_pool = [
        ["VS Code", "Git"], ["Vim", "Docker"],
        ["IntelliJ", "Gradle"], ["Jupyter", "pandas"],
        ["Figma", "Sketch"], ["Terraform", "AWS CLI"],
    ]

    personas = []
    for i in range(count):
        # Ensure dimension diversity
        ground_truth: dict = {}

        # demographics (18/20 have name)
        if i < 18:
            ground_truth["demographics"] = {"name": names[i]}
        else:
            ground_truth["demographics"] = {}

        # background (16/20 have occupation, 8/20 have expertise_areas)
        bg: dict = {}
        if i < 16:
            bg["occupation"] = occupations[i]
        if i < 8:
            bg["expertise_areas"] = random.sample(
                ["programming", "design", "analytics", "security", "testing", "management"],
                k=random.randint(1, 3),
            )
        if i < 6:
            bg["current_projects"] = [f"project_{chr(65 + i)}"]
        if bg:
            ground_truth["background"] = bg

        # interests (18/20 topics, 14/20 hobbies)
        interests: dict = {}
        if i < 18:
            interests["topics"] = topics_pool[i % len(topics_pool)]
        if i < 14:
            interests["hobbies"] = hobbies_pool[i % len(hobbies_pool)]
        if interests:
            ground_truth["interests"] = interests

        # preferences (6/20)
        if i < 6:
            ground_truth["preferences"] = {
                "work_style": random.choice(["structured", "flexible"]),
                "tools": tools_pool[i % len(tools_pool)],
            }

        # Generate 5-turn dialogue revealing the profile info
        dialogue = _generate_dialogue(ground_truth, i)

        persona = {
            "source": f"PersonaExt:synthetic_{i + 1:03d}",
            "ground_truth": ground_truth,
            "dialogue_turns": dialogue,
        }
        personas.append(persona)

    return personas


def _generate_dialogue(gt: dict, idx: int) -> list[dict]:
    """Generate a 5-turn dialogue that naturally reveals profile fields."""
    turns = []
    demo = gt.get("demographics", {})
    bg = gt.get("background", {})
    interests = gt.get("interests", {})
    prefs = gt.get("preferences", {})

    # Turn 1: name + greeting
    name = demo.get("name", "friend")
    turns.append({"role": "user", "content": f"Hi! My name is {name}. Nice to meet you!"})
    turns.append({"role": "assistant", "content": f"Nice to meet you too, {name}! How can I help you today?"})

    # Turn 2: occupation
    occ = bg.get("occupation")
    if occ:
        turns.append({"role": "user", "content": f"I work as a {occ}. I've been doing this for a few years now."})
        turns.append({"role": "assistant", "content": f"That's great! A {occ} — sounds interesting. What kind of things do you work on?"})

    # Turn 3: expertise / project
    expertise = bg.get("expertise_areas", [])
    projects = bg.get("current_projects", [])
    if expertise:
        turns.append({"role": "user", "content": f"I mostly work with {', '.join(expertise)}. {f'Currently working on {projects[0]}.' if projects else ''}"})
        turns.append({"role": "assistant", "content": "Got it! Sounds like you have a solid skill set."})

    # Turn 4: interests / hobbies
    topics = interests.get("topics", [])
    hobbies = interests.get("hobbies", [])
    if hobbies:
        turns.append({"role": "user", "content": f"Outside of work I enjoy {', '.join(hobbies)}. {f'I also like reading about {topics[0]}.' if topics else ''}"})
        turns.append({"role": "assistant", "content": "Those are some great hobbies! Always good to have a healthy work-life balance."})

    # Turn 5: preferences
    work_style = prefs.get("work_style")
    tools = prefs.get("tools", [])
    if work_style:
        turns.append({"role": "user", "content": f"I prefer a {work_style} approach to work. {f'My go-to tools are {tools[0]} and {tools[1]}.' if tools else ''}"})
        turns.append({"role": "assistant", "content": f"Noted! I'll keep your {work_style} preference in mind."})
    elif topics:
        # Fallback: mention remaining topics
        remaining = topics[1:] if len(topics) > 1 else topics
        turns.append({"role": "user", "content": f"By the way, I'm also interested in {', '.join(remaining)}."})
        turns.append({"role": "assistant", "content": "I'll keep that in mind!"})

    # Ensure at least 5 user turns (pad if needed)
    while len([t for t in turns if t["role"] == "user"]) < 5:
        turns.append({"role": "user", "content": "Thanks, that's helpful!"})
        turns.append({"role": "assistant", "content": "You're welcome! Let me know if you need anything else."})

    return turns


def _download_personaext(raw_dir: Path) -> bool:
    """Attempt to download PersonaExt data from HuggingFace."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        logger.info("Downloading PersonaExt from HuggingFace...")
        ds = load_dataset("giladiyoel/persona-ext", split="train")
        output_path = raw_dir / "personaext.json"
        ds.to_json(str(output_path))
        logger.info("Downloaded %d records to %s", len(ds), output_path)
        return True
    except ImportError:
        logger.warning(
            "`datasets` package not available. "
            "Install with: pip install datasets"
        )
    except Exception as e:
        logger.warning("Failed to download PersonaExt: %s", e)

    return False


def _convert_personaext(raw_dir: Path) -> list[dict]:
    """Convert raw PersonaExt data to SPB persona format."""
    raw_path = raw_dir / "personaext.json"
    if not raw_path.exists():
        return []

    with open(raw_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    # Load mapping
    with open(MAPPING_PATH, encoding="utf-8") as f:
        mapping = json.load(f)

    predicate_map = mapping.get("predicate_mapping", {})
    excluded = set(mapping.get("excluded_predicates", []))

    personas = []
    for idx, entry in enumerate(raw_data):
        triplets = entry.get("triplets", [])
        if not triplets:
            continue

        gt: dict = {}
        for triplet in triplets:
            predicate = triplet.get("predicate", "")
            if predicate in excluded:
                continue

            pmap = predicate_map.get(predicate)
            if not pmap:
                continue

            dim = pmap["target_dimension"]
            field = pmap["target_field"]
            ftype = pmap["field_type"]
            value = triplet.get("object", "")

            if dim not in gt:
                gt[dim] = {}

            if ftype == "list":
                if field not in gt[dim]:
                    gt[dim][field] = []
                gt[dim][field].append(value)
            else:
                gt[dim][field] = value

        if not gt:
            continue

        personas.append({
            "source": f"PersonaExt:{entry.get('id', idx)}",
            "ground_truth": gt,
            "dialogue_turns": entry.get("dialogue", []),
        })

    return personas


def _score_persona(persona: dict) -> int:
    """Score a persona against selection criteria (0–10)."""
    gt = persona.get("ground_truth", {})
    score = 0

    # Field density (0–3)
    total_fields = sum(len(v) for v in gt.values() if isinstance(v, dict))
    if total_fields >= 5:
        score += 3
    elif total_fields >= 3:
        score += 2
    elif total_fields >= 1:
        score += 1

    # Dimension diversity (0–3)
    n_dims = len([k for k, v in gt.items() if isinstance(v, dict) and v])
    if n_dims >= 3:
        score += 3
    elif n_dims == 2:
        score += 2
    elif n_dims == 1:
        score += 1

    # List fields (0–2)
    has_list = any(
        isinstance(v, list)
        for d in gt.values() if isinstance(d, dict)
        for v in d.values()
    )
    if has_list:
        score += 2

    # Turn distribution (0–1)
    dialogue = persona.get("dialogue_turns", [])
    user_turns = [t for t in dialogue if t.get("role") == "user"]
    if len(user_turns) >= 3:
        score += 1

    # Lexical heterogeneity (0–1) — always award for synthetic data
    score += 1

    return score


def build(count: int = 20, skip_download: bool = False) -> None:
    """Build the SPB evaluation set."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PERSONAS_DIR.mkdir(parents=True, exist_ok=True)

    # Try to load or download PersonaExt
    personas = []

    if not skip_download:
        if _download_personaext(RAW_DIR):
            personas = _convert_personaext(RAW_DIR)
            logger.info("Converted %d personas from PersonaExt", len(personas))

    # Fallback to synthetic data if no PersonaExt data
    if not personas:
        logger.info("Using synthetic persona data (PersonaExt unavailable)")
        personas = _generate_synthetic_personas(count)

    # Score and rank
    scored = [(p, _score_persona(p)) for p in personas]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Select top N
    selected = scored[:count]
    logger.info("Selected %d personas (scores: %s)", len(selected), [s for _, s in selected])

    # Write output
    for i, (persona, score) in enumerate(selected):
        persona_id = f"pe_{i + 1:03d}"
        output = {
            "source": persona.get("source", f"synthetic_{i + 1}"),
            "score": score,
            "ground_truth": persona.get("ground_truth", {}),
            "dialogue_turns": persona.get("dialogue_turns", []),
        }
        out_path = PERSONAS_DIR / f"{persona_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("  Written %s (%d fields, score=%d)",
                     persona_id,
                     sum(len(v) for v in output["ground_truth"].values() if isinstance(v, dict)),
                     score)

    logger.info("Done! %d personas written to %s", len(selected), PERSONAS_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build SPB evaluation task set")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip PersonaExt download, use synthetic data")
    parser.add_argument("--count", type=int, default=20,
                        help="Number of personas to generate (default: 20)")
    args = parser.parse_args()

    build(count=args.count, skip_download=args.skip_download)
