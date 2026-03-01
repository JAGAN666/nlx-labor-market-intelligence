"""
Step 3: Skill Recommender Demo
===============================
Input:  A list of skills you already have
Output: Top-K recommended "next skills to learn"

Usage:
  python3 3_recommend.py
  python3 3_recommend.py --skills "Microsoft Excel,communicate with customers" --top_k 10
"""

import pickle
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ──────────────────────────────────────────────────────────────────
IN_EMBEDDINGS   = "skill_embeddings.npy"
IN_SKILL_INDEX  = "skill_index.pkl"
DEFAULT_TOP_K   = 10
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    embeddings = np.load(IN_EMBEDDINGS)
    with open(IN_SKILL_INDEX, "rb") as f:
        skill_to_idx = pickle.load(f)
    idx_to_skill = {v: k for k, v in skill_to_idx.items()}
    return embeddings, skill_to_idx, idx_to_skill

def find_closest_skill(query: str, skill_to_idx: dict) -> str | None:
    """Fuzzy match: find the closest skill name in our index."""
    query = query.lower().strip()
    # Exact match
    for skill in skill_to_idx:
        if skill.lower() == query:
            return skill
    # Partial match
    candidates = [s for s in skill_to_idx if query in s.lower() or s.lower() in query]
    if candidates:
        return candidates[0]
    return None

def recommend(input_skills: list[str], top_k: int = DEFAULT_TOP_K):
    embeddings, skill_to_idx, idx_to_skill = load_model()
    N = len(embeddings)

    # ── resolve input skills ──────────────────────────────────────────────
    resolved, unresolved = [], []
    for s in input_skills:
        match = find_closest_skill(s, skill_to_idx)
        if match:
            resolved.append(match)
            if match.lower() != s.lower():
                print(f"   ℹ️  '{s}' → matched as '{match}'")
        else:
            unresolved.append(s)
            print(f"   ⚠️  '{s}' not found in skill index. Skipping.")

    if not resolved:
        print("\n❌ No valid skills found. Please try different skill names.")
        return

    # ── compute profile vector ────────────────────────────────────────────
    idxs = [skill_to_idx[s] for s in resolved]
    profile_vec = embeddings[idxs].mean(axis=0, keepdims=True)

    # ── cosine similarity against all skills ──────────────────────────────
    sims = cosine_similarity(profile_vec, embeddings)[0]

    # Mask out input skills
    for i in idxs:
        sims[i] = -1.0

    # Top-K
    top_idx = np.argsort(sims)[::-1][:top_k]

    # ── Display ───────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f" 🎓 SKILL RECOMMENDER — Next Skills to Learn")
    print(f"{'═'*55}")
    print(f"\n 📌 Your current skills:")
    for s in resolved:
        print(f"   ✓ {s}")

    print(f"\n 🚀 Recommended next skills (top {top_k}):")
    print(f"   {'#':<4} {'Skill':<40} {'Score':>6}")
    print(f"   {'-'*52}")
    for rank, idx in enumerate(top_idx, 1):
        skill = idx_to_skill[idx]
        score = sims[idx]
        print(f"   {rank:<4} {skill:<40} {score:.3f}")

    print(f"\n{'═'*55}\n")
    return [(idx_to_skill[i], float(sims[i])) for i in top_idx]

def interactive_mode(embeddings, skill_to_idx, idx_to_skill):
    """Browse available skills and get recommendations interactively."""
    print("\n" + "═"*55)
    print(" 🎮 INTERACTIVE SKILL RECOMMENDER")
    print("═"*55)
    print(" Type comma-separated skills, 'list' to browse, or 'q' to quit")
    print("═"*55 + "\n")

    while True:
        user_in = input(" Enter your skills: ").strip()
        if user_in.lower() == 'q':
            break
        elif user_in.lower() == 'list':
            all_skills = sorted(skill_to_idx.keys())
            for i, s in enumerate(all_skills[:50], 1):
                print(f"   {i:>3}. {s}")
            print(f"   ... and {len(all_skills)-50:,} more")
        else:
            input_skills = [s.strip() for s in user_in.split(",") if s.strip()]
            recommend(input_skills, top_k=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skill Recommender")
    parser.add_argument("--skills", type=str, default=None,
                        help="Comma-separated list of skills you have")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help="Number of recommendations to return")
    args = parser.parse_args()

    embeddings, skill_to_idx, idx_to_skill = load_model()

    if args.skills:
        input_skills = [s.strip() for s in args.skills.split(",") if s.strip()]
        recommend(input_skills, top_k=args.top_k)
    else:
        interactive_mode(embeddings, skill_to_idx, idx_to_skill)
