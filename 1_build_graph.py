"""
Step 1: Build Skill-Skill Co-occurrence Graph
=============================================
Reads colorado_processed.csv and constructs:
  - A bipartite graph: ResearchID <-> TaxonomySkill
  - A pruned skill-skill co-occurrence graph (edge weight = shared jobs)
Saves the result as skill_graph.pkl
"""

import pandas as pd
import networkx as nx
import pickle
from itertools import combinations
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH        = "colorado_processed.csv"
OUT_GRAPH        = "skill_graph.pkl"
OUT_SKILLS_LIST  = "skill_list.pkl"
MIN_SKILL_JOBS   = 5   # skill must appear in >= 5 jobs
MIN_EDGE_WEIGHT  = 3   # pair must co-occur in >= 3 jobs
# ─────────────────────────────────────────────────────────────────────────────

print("📥 Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"   Loaded {len(df):,} rows | {df['Research ID'].nunique():,} jobs | {df['Taxonomy Skill'].nunique():,} skills")

# ── 1. Filter rare skills ─────────────────────────────────────────────────
skill_counts = df["Taxonomy Skill"].value_counts()
valid_skills  = set(skill_counts[skill_counts >= MIN_SKILL_JOBS].index)
df = df[df["Taxonomy Skill"].isin(valid_skills)]
print(f"\n✅ After filtering (>= {MIN_SKILL_JOBS} jobs): {len(valid_skills):,} unique skills kept")

# ── 2. Build job → skills mapping ─────────────────────────────────────────
job_skills = df.groupby("Research ID")["Taxonomy Skill"].apply(list).to_dict()
print(f"   Jobs with valid skills: {len(job_skills):,}")

# ── 3. Build co-occurrence edge weights ──────────────────────────────────
print("\n🔗 Building skill co-occurrence edges...")
edge_weights = {}
for rid, skills in tqdm(job_skills.items(), desc="Jobs processed"):
    unique_skills = list(set(skills))
    for s1, s2 in combinations(unique_skills, 2):
        key = tuple(sorted([s1, s2]))
        edge_weights[key] = edge_weights.get(key, 0) + 1

print(f"   Raw edges before pruning: {len(edge_weights):,}")

# ── 4. Prune weak edges & build NetworkX graph ────────────────────────────
G = nx.Graph()

for (s1, s2), weight in edge_weights.items():
    if weight >= MIN_EDGE_WEIGHT:
        G.add_edge(s1, s2, weight=weight)

print(f"\n✅ After pruning (weight >= {MIN_EDGE_WEIGHT})")
print(f"   Nodes (skills): {G.number_of_nodes():,}")
print(f"   Edges:          {G.number_of_edges():,}")
print(f"   Avg degree:     {sum(dict(G.degree()).values()) / G.number_of_nodes():.1f}")

# ── 5. Save ──────────────────────────────────────────────────────────────
with open(OUT_GRAPH, "wb") as f:
    pickle.dump(G, f)

skill_list = list(G.nodes())
with open(OUT_SKILLS_LIST, "wb") as f:
    pickle.dump(skill_list, f)

print(f"\n💾 Saved graph → {OUT_GRAPH}")
print(f"💾 Saved skill list → {OUT_SKILLS_LIST}")
print("\n✨ Step 1 complete! Run '2_train_embeddings.py' next.")
