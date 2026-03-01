"""
Step 4: Workforce Graph Intelligence Analytics Engine
=====================================================
5 Modules:
  1. Skill Ecosystem Detection   (Louvain community detection)
  2. Gatekeeper Skill Ranking    (betweenness + eigenvector centrality)
  3. Career Mobility Simulation  (cluster-level shortest path)
  4. Workforce Inequality        (cluster isolation + Gini coefficient)
  5. Emerging Skill Forecast     (structural position classification)

Outputs:
  wgi_data.pkl  — single dict with all results, loaded by app.py
"""

import pickle
import numpy as np
import networkx as nx
import pandas as pd
from collections import defaultdict
import community as community_louvain   # python-louvain
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  Workforce Graph Intelligence — Analytics Engine")
print("=" * 60)

# ── Load graph ───────────────────────────────────────────────────────────────
print("\n📥 Loading graph...")
with open("skill_graph.pkl", "rb") as f:
    G = pickle.load(f)
with open("skill_index.pkl", "rb") as f:
    skill_to_idx = pickle.load(f)

N = G.number_of_nodes()
print(f"   Nodes: {N:,} | Edges: {G.number_of_edges():,}")

# ═══════════════════════════════════════════════════════════════
# MODULE 1 — Skill Ecosystem Detection (Louvain)
# ═══════════════════════════════════════════════════════════════
print("\n[ M1 ] 🌐 Skill Ecosystem Detection...")

partition = community_louvain.best_partition(G, weight="weight", random_state=42)
# partition: {skill: cluster_id}

# Remap cluster IDs to 0-indexed
raw_ids = sorted(set(partition.values()))
remap = {old: new for new, old in enumerate(raw_ids)}
partition = {skill: remap[cid] for skill, cid in partition.items()}

n_clusters = len(set(partition.values()))
print(f"   Found {n_clusters} clusters")

# Label each cluster by its top-5 skills (by in-cluster degree)
cluster_to_skills = defaultdict(list)
for skill, cid in partition.items():
    cluster_to_skills[cid].append(skill)

cluster_labels = {}
cluster_profiles = {}
for cid, members in cluster_to_skills.items():
    subgraph = G.subgraph(members)
    # top skills by within-cluster degree
    deg_within = dict(subgraph.degree(weight="weight"))
    top_skills  = sorted(deg_within, key=lambda s: deg_within[s], reverse=True)[:5]
    label = " & ".join(top_skills[:2])
    cluster_labels[cid] = label
    cluster_profiles[cid] = {
        "label":      label,
        "top_skills": top_skills,
        "size":       len(members),
        "members":    members,
    }
    print(f"   Cluster {cid:>2} ({len(members):>4} skills): {label}")

# ═══════════════════════════════════════════════════════════════
# MODULE 2 — Gatekeeper Skill Ranking
# ═══════════════════════════════════════════════════════════════
print("\n[ M2 ] 🔑 Computing Gatekeeper Skills...")

print("   Betweenness centrality (may take ~30s)...")
betweenness  = nx.betweenness_centrality(G, weight="weight", normalized=True, k=500)
print("   Eigenvector centrality...")
try:
    eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
except:
    eigenvector = nx.degree_centrality(G)

degree_cent  = nx.degree_centrality(G)

# Normalize all to [0, 1]
def norm01(d):
    vals = np.array(list(d.values()))
    mn, mx = vals.min(), vals.max()
    if mx == mn: return {k: 0.0 for k in d}
    return {k: float((v - mn) / (mx - mn)) for k, v in d.items()}

btw_n   = norm01(betweenness)
eig_n   = norm01(eigenvector)
deg_n   = norm01(degree_cent)

gatekeeper_scores = {}
for skill in G.nodes():
    # High betweenness = bridge; high eigenvector = important neighbors
    score = 0.5 * btw_n[skill] + 0.35 * eig_n[skill] + 0.15 * deg_n[skill]
    gatekeeper_scores[skill] = {
        "score":       round(score, 4),
        "betweenness": round(btw_n[skill], 4),
        "eigenvector": round(eig_n[skill], 4),
        "degree":      round(deg_n[skill], 4),
        "cluster":     partition.get(skill, -1),
        "bridges":     []  # filled below
    }

# Find which clusters each gatekeeper bridges
for skill in G.nodes():
    own_cluster = partition.get(skill, -1)
    neighbor_clusters = set()
    for nb in G.neighbors(skill):
        nc = partition.get(nb, -1)
        if nc != own_cluster:
            neighbor_clusters.add(nc)
    gatekeeper_scores[skill]["bridges"] = sorted(neighbor_clusters)

top_gatekeepers = sorted(gatekeeper_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:30]
print(f"   Top 5 gatekeepers:")
for skill, data in top_gatekeepers[:5]:
    print(f"     {skill:<40} score={data['score']:.3f}  bridges={len(data['bridges'])} clusters")

# ═══════════════════════════════════════════════════════════════
# MODULE 3 — Career Mobility Simulation
# ═══════════════════════════════════════════════════════════════
print("\n[ M3 ] 🗺️  Building Career Mobility Graph...")

# Build cluster-level graph
C = nx.Graph()
for cid in cluster_profiles:
    C.add_node(cid, **cluster_profiles[cid])

# Inter-cluster edges: weight = # cross-cluster edges, bridge skills = list of connector skills
inter_edges = defaultdict(lambda: {"weight": 0, "bridge_skills": []})
for u, v, data in G.edges(data=True):
    cu, cv = partition.get(u, -1), partition.get(v, -1)
    if cu != cv:
        key = tuple(sorted([cu, cv]))
        inter_edges[key]["weight"] += data.get("weight", 1)
        inter_edges[key]["bridge_skills"].append((u, v, data.get("weight", 1)))

for (cu, cv), info in inter_edges.items():
    # Transition difficulty: lower weight = harder (fewer shared jobs)
    difficulty = round(1.0 / (1.0 + np.log1p(info["weight"])), 3)
    # Top bridge skills for this pair
    top_bridges = sorted(info["bridge_skills"], key=lambda x: x[2], reverse=True)[:5]
    C.add_edge(cu, cv,
               weight=info["weight"],
               difficulty=difficulty,
               bridge_skills=top_bridges)

print(f"   Cluster graph: {C.number_of_nodes()} nodes, {C.number_of_edges()} edges")

# ═══════════════════════════════════════════════════════════════
# MODULE 4 — Workforce Inequality Detection
# ═══════════════════════════════════════════════════════════════
print("\n[ M4 ] ⚖️  Computing Workforce Inequality...")

inequality_report = {}
all_degrees = [d for _, d in G.degree(weight="weight")]
all_degrees_arr = np.array(sorted(all_degrees))

# Gini coefficient of degree distribution
def gini(arr):
    arr = np.sort(arr.astype(float))
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum()) - (n + 1) / n))

global_gini = round(gini(all_degrees_arr), 4)
print(f"   Global skill concentration Gini: {global_gini:.3f}")

for cid, profile in cluster_profiles.items():
    members = profile["members"]
    subgraph = G.subgraph(members)

    # Intra-cluster edge weight sum
    intra_w = sum(d.get("weight", 1) for _, _, d in subgraph.edges(data=True))

    # Inter-cluster edge weight sum
    inter_w = 0
    bridge_count = 0
    for skill in members:
        for nb in G.neighbors(skill):
            if partition.get(nb, -1) != cid:
                inter_w += G[skill][nb].get("weight", 1)
                bridge_count += 1

    total_w = intra_w + inter_w + 1e-9
    isolation = round(intra_w / total_w, 4)

    # Degree distribution Gini within cluster
    cluster_degrees = np.array([d for _, d in subgraph.degree(weight="weight")])
    cluster_gini = round(gini(cluster_degrees), 4) if len(cluster_degrees) > 1 else 0.0

    # Locked = high isolation AND few bridge skills
    locked = isolation > 0.75 and bridge_count < 20

    inequality_report[cid] = {
        "label":          profile["label"],
        "size":           profile["size"],
        "isolation":      isolation,
        "gini":           cluster_gini,
        "bridge_count":   bridge_count,
        "locked":         locked,
        "top_skills":     profile["top_skills"][:3],
    }
    flag = "🔒 LOCKED" if locked else ""
    print(f"   Cluster {cid:>2}: isolation={isolation:.2f}  gini={cluster_gini:.2f}  bridges={bridge_count:>4}  {flag}")

# ═══════════════════════════════════════════════════════════════
# MODULE 5 — Emerging Skill Forecast
# ═══════════════════════════════════════════════════════════════
print("\n[ M5 ] 📈 Classifying Skill Trajectories...")

skill_forecast = {}
for skill in G.nodes():
    b = btw_n[skill]    # bridge importance
    e = eig_n[skill]    # neighbor quality
    d = deg_n[skill]    # raw connections

    # Classification rules:
    if b > 0.5 and d < 0.4:
        tag = "Rising"       # high bridge, low presence = emerging connector
        desc = "Structurally important but not yet dominant. On the rise."
    elif d > 0.6 and e > 0.5:
        tag = "Dominant"     # high degree + important neighbors = pillar skill
        desc = "Core pillar of the workforce. High demand, well-connected."
    elif b < 0.1 and d < 0.2:
        tag = "Niche"        # low connectivity = specialized
        desc = "Highly specialized. Deep domain but limited cross-cluster reach."
    else:
        tag = "Established"  # middle ground — stable
        desc = "Stable mid-tier skill with consistent demand."

    skill_forecast[skill] = {
        "tag":   tag,
        "desc":  desc,
        "btw":   round(b, 4),
        "eig":   round(e, 4),
        "deg":   round(d, 4),
        "cluster": partition.get(skill, -1),
    }

tag_counts = defaultdict(int)
for v in skill_forecast.values():
    tag_counts[v["tag"]] += 1
for tag, count in sorted(tag_counts.items()):
    print(f"   {tag:<12}: {count:>4} skills")

# ═══════════════════════════════════════════════════════════════
# Save all results
# ═══════════════════════════════════════════════════════════════
print("\n💾 Saving WGI data...")

# Build node list for force graph visualization
node_data = []
for skill in G.nodes():
    node_data.append({
        "id":      skill,
        "cluster": partition.get(skill, -1),
        "degree":  G.degree(skill, weight="weight"),
        "gatekeeper_score": gatekeeper_scores[skill]["score"],
        "forecast_tag": skill_forecast[skill]["tag"],
    })

# Build edge list (sample top 5000 edges by weight for viz performance)
all_edges = sorted(
    [{"source": u, "target": v, "weight": d.get("weight", 1)}
     for u, v, d in G.edges(data=True)],
    key=lambda x: x["weight"], reverse=True
)[:5000]

wgi = {
    "partition":          partition,
    "cluster_profiles":   cluster_profiles,
    "cluster_labels":     cluster_labels,
    "gatekeeper_scores":  gatekeeper_scores,
    "top_gatekeepers":    [(s, d) for s, d in top_gatekeepers],
    "cluster_graph":      C,
    "inequality_report":  inequality_report,
    "global_gini":        global_gini,
    "skill_forecast":     skill_forecast,
    "node_data":          node_data,
    "edge_data":          all_edges,
    "n_clusters":         n_clusters,
}

with open("wgi_data.pkl", "wb") as f:
    pickle.dump(wgi, f)

print("✅ Saved → wgi_data.pkl")
print("\n" + "=" * 60)
print("  ✨ WGI Analytics Engine Complete!")
print(f"     {n_clusters} ecosystems | {len(top_gatekeepers)} gatekeepers")
print(f"     Global Gini: {global_gini:.3f}")
print("=" * 60)
