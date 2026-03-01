"""
Step 2 (v2): Graph NLP Embeddings — SBERT + Graph Message Passing
=================================================================
Approach (as recommended by Margaret):
  1. Use Sentence-BERT (all-MiniLM-L6-v2) to encode each skill's TEXT into a
     dense semantic vector — this captures NLP meaning of the skill name/description
  2. Run graph-aware propagation (spectral smoothing) on the co-occurrence graph
     to blend each skill's vector with its neighbors' — this brings in structural info
  3. The result: embeddings that are BOTH semantically meaningful AND graph-aware

No Node2Vec. No anonymous random walks. Pure NLP + Graph.

Saves:
  - skill_embeddings.npy  (N_skills × 384 matrix, L2-normalized)
  - skill_index.pkl       (skill_name → row index)
"""

import pickle
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.sparse import diags
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
IN_GRAPH        = "skill_graph.pkl"
IN_SKILLS_LIST  = "skill_list.pkl"
IN_DATA         = "colorado_processed.csv"
OUT_EMBEDDINGS  = "skill_embeddings.npy"
OUT_SKILL_INDEX = "skill_index.pkl"

SBERT_MODEL     = "all-MiniLM-L6-v2"  # 384-dim, fast, great quality
PROPAGATION_STEPS = 2                  # how many hops of graph smoothing
ALPHA             = 0.5                # mix: 0=only SBERT, 1=only neighbors
# ─────────────────────────────────────────────────────────────────────────────

print("📥 Loading graph and data...")
with open(IN_GRAPH, "rb") as f:
    G = pickle.load(f)
with open(IN_SKILLS_LIST, "rb") as f:
    skills = pickle.load(f)

# Load taxonomy descriptions for richer text encoding
df = pd.read_csv(IN_DATA)
skill_desc = df.dropna(subset=["Taxonomy Description"]) \
               .drop_duplicates("Taxonomy Skill") \
               .set_index("Taxonomy Skill")["Taxonomy Description"] \
               .to_dict()

N = len(skills)
skill_to_idx = {s: i for i, s in enumerate(skills)}
print(f"   Skills: {N:,} | Graph edges: {G.number_of_edges():,}")

# ── Step 1: Build text strings for each skill ─────────────────────────────
print("\n📝 Preparing skill text representations...")
texts = []
for skill in skills:
    desc = skill_desc.get(skill, "")
    # Combine skill name + description for richer NLP signal
    text = f"{skill}. {desc}" if desc else skill
    texts.append(text)

print(f"   Skills with descriptions: {sum(1 for s in skills if s in skill_desc):,} / {N:,}")

# ── Step 2: Encode with Sentence-BERT ─────────────────────────────────────
print(f"\n🤖 Loading Sentence-BERT model: {SBERT_MODEL} ...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(SBERT_MODEL)

print(f"   Encoding {N:,} skill texts (batched)...")
sbert_embeddings = model.encode(
    texts,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True
)
sbert_embeddings = normalize(sbert_embeddings, norm="l2")
print(f"   SBERT embedding shape: {sbert_embeddings.shape}")

# ── Step 3: Build normalized adjacency for graph propagation ──────────────
print(f"\n🔗 Building graph propagation matrix ({PROPAGATION_STEPS} hops)...")

# Build adjacency matrix from graph edges (weighted)
from scipy.sparse import lil_matrix
adj = lil_matrix((N, N), dtype=np.float32)
for u, v, data in G.edges(data=True):
    if u in skill_to_idx and v in skill_to_idx:
        i, j = skill_to_idx[u], skill_to_idx[v]
        w = float(data.get("weight", 1))
        adj[i, j] = w
        adj[j, i] = w

adj = adj.tocsr()

# Symmetric normalized: D^{-1/2} A D^{-1/2}  (standard GCN normalization)
deg = np.array(adj.sum(axis=1)).flatten()
deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
D_inv_sqrt = diags(deg_inv_sqrt)
norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt  # sparse symmetric normalized

# ── Step 4: Graph propagation (message passing) ───────────────────────────
# H^{(t+1)} = (1-α)*H^{(0)} + α * A_norm * H^{(t)}
embeddings = sbert_embeddings.copy()
for step in range(PROPAGATION_STEPS):
    neighbor_msg = norm_adj @ embeddings  # each node aggregates neighbors
    embeddings = (1 - ALPHA) * sbert_embeddings + ALPHA * neighbor_msg
    embeddings = normalize(embeddings, norm="l2")
    print(f"   Propagation step {step+1}/{PROPAGATION_STEPS} done")

print(f"   Final embedding shape: {embeddings.shape}")

# ── Save ─────────────────────────────────────────────────────────────────
np.save(OUT_EMBEDDINGS, embeddings)
with open(OUT_SKILL_INDEX, "wb") as f:
    pickle.dump(skill_to_idx, f)

# Quick sanity: nearest neighbors of "Microsoft Excel"
target = "Microsoft Excel"
if target in skill_to_idx:
    from sklearn.metrics.pairwise import cosine_similarity
    idx   = skill_to_idx[target]
    sims  = cosine_similarity(embeddings[idx:idx+1], embeddings)[0]
    sims[idx] = -1
    top5  = np.argsort(sims)[::-1][:5]
    idx_to_skill = {v: k for k, v in skill_to_idx.items()}
    print(f"\n🔍 Sanity check — neighbors of '{target}':")
    for i in top5:
        print(f"   {idx_to_skill[i]:<40} {sims[i]:.3f}")

print(f"\n💾 Saved embeddings → {OUT_EMBEDDINGS}")
print(f"💾 Saved skill index → {OUT_SKILL_INDEX}")
print("\n✨ Step 2 complete! Graph NLP embeddings ready.")
print(f"   Method: Sentence-BERT ({SBERT_MODEL}) + {PROPAGATION_STEPS}-hop graph propagation")
