"""
Pre-compute graph node positions using NetworkX spring layout.
Saves positions to graph_positions.pkl so the browser gets
static coordinates — no live D3 force simulation needed.
"""
import pickle
import numpy as np
import networkx as nx

print("Loading graph...")
with open("skill_graph.pkl", "rb") as f:
    G = pickle.load(f)

with open("wgi_data.pkl", "rb") as f:
    wgi = pickle.load(f)

N = G.number_of_nodes()
print(f"Nodes: {N}")

# ── Use kamada_kawai on a subgraph of top-degree nodes for speed,
#    then place remaining nodes proportionally ─────────────────────
print("Computing layout (spring)...")

# Sample top 800 nodes by degree for the seed layout
degree_sorted = sorted(G.nodes(), key=lambda n: G.degree(n, weight='weight'), reverse=True)

# Build a subgraph of top 800 nodes for fast seed layout
seed_nodes = degree_sorted[:800]
H = G.subgraph(seed_nodes)

seed_pos = nx.spring_layout(
    H,
    weight='weight',
    k=2.0 / (len(H) ** 0.5),
    iterations=80,
    seed=42,
    scale=900
)

# For remaining nodes, assign a position near their most-connected seed-graph neighbor
node_pos = dict(seed_pos)
remaining = [n for n in G.nodes() if n not in seed_pos]

partition = wgi["partition"]
# cluster centers from seed nodes
cluster_centers = {}
for n, (x, y) in seed_pos.items():
    c = partition.get(n, 0)
    if c not in cluster_centers:
        cluster_centers[c] = [0.0, 0.0, 0]
    cluster_centers[c][0] += x
    cluster_centers[c][1] += y
    cluster_centers[c][2] += 1

for c in cluster_centers:
    total = cluster_centers[c][2]
    cluster_centers[c] = (cluster_centers[c][0]/total, cluster_centers[c][1]/total)

rng = np.random.RandomState(42)
for n in remaining:
    c = partition.get(n, 0)
    if c in cluster_centers:
        cx, cy = cluster_centers[c]
        jitter = rng.uniform(-80, 80, 2)
        node_pos[n] = (cx + jitter[0], cy + jitter[1])
    else:
        node_pos[n] = (rng.uniform(-900, 900), rng.uniform(-900, 900))

print(f"Positioned {len(node_pos)} nodes")

# Normalize to [50, canvas-50]
xs = [p[0] for p in node_pos.values()]
ys = [p[1] for p in node_pos.values()]
xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys), max(ys)

W, H_canvas = 1200, 700  # reference canvas size
PAD = 60

def norm(v, vmin, vmax, omin, omax):
    if vmax == vmin: return (omin + omax) / 2
    return omin + (v - vmin) / (vmax - vmin) * (omax - omin)

normalized = {
    n: (
        round(norm(xy[0], xmin, xmax, PAD, W - PAD), 2),
        round(norm(xy[1], ymin, ymax, PAD, H_canvas - PAD), 2)
    )
    for n, xy in node_pos.items()
}

with open("graph_positions.pkl", "wb") as f:
    pickle.dump(normalized, f)

print(f"✅ Saved graph_positions.pkl ({len(normalized)} nodes)")
