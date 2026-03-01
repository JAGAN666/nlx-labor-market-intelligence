import pandas as pd
import networkx as nx
import community as community_louvain
import pickle
from collections import defaultdict
from itertools import combinations

print("=== Workforce Graph Intelligence Builder ===\n")

# Skill categories for node styling
SKILL_CATEGORIES = {
    "Communication": ["communicat", "customer service", "presentation", "writing", "verbal", "documentation", "report"],
    "Technical / IT": ["software", "python", "data", "computer", "programming", "database", "sql", "cloud", "aws", "azure", "excel", "microsoft"],
    "Leadership": ["leadership", "management", "supervise", "team lead", "coordinate", "direct", "train staff"],
    "Clinical / Medical": ["patient", "nursing", "medical", "clinical", "diagnos", "therapy", "medication", "cna", "rn", "surgery"],
    "Physical / Labor": ["lift", "equipment", "maintenance", "machinery", "construction", "operate", "installation", "physical"],
    "Analytical": ["analys", "research", "problem-solv", "planning", "strategy", "forecast", "evaluat"],
    "Compliance / Safety": ["compliance", "safety", "regulation", "osha", "license", "certif", "legal", "quality control"],
    "Customer Facing": ["sales", "retail", "client", "customer", "advise", "service represent"],
}

def categorize_skill(s):
    s = str(s).lower()
    for cat, keywords in SKILL_CATEGORIES.items():
        if any(kw in s for kw in keywords):
            return cat
    return "Other"

print("1. Loading specific skill extractions...")
skills = pd.read_csv("colorado_processed.csv", dtype={"Research ID": str})
skills.columns = ["job_id","raw_skill","taxonomy_skill","taxonomy_desc","taxonomy_source","correlation"]
skills = skills[skills["correlation"] >= 0.5]

print("2. Filtering to top skills for clean visualization...")
skill_counts = skills["taxonomy_skill"].value_counts()
# Take top 800 most frequent skills to ensure graph is dense but performant
top_skills = set(skill_counts.head(800).index)

skills_top = skills[skills["taxonomy_skill"].isin(top_skills)]

print("3. Building co-occurrence network edges...")
# Group by job to find co-occurring skills
job_groups = skills_top.groupby("job_id")["taxonomy_skill"].unique()

edge_weights = defaultdict(int)
for job_skills in job_groups:
    job_skills_sorted = sorted(job_skills)
    for u, v in combinations(job_skills_sorted, 2):
        edge_weights[(u, v)] += 1

G = nx.Graph()
# Only add edges that occur at least 3 times to filter noise
for (u, v), w in edge_weights.items():
    if w >= 3:
         G.add_edge(u, v, weight=w)

print(f"  Raw graph size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Extract Largest Connected Component
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()
print(f"  LCC size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Inverse weight for shortest paths and betweenness 
# (high co-occurrence = short distance)
for u, v, d in G.edges(data=True):
    d['distance'] = 1.0 / d['weight']

# Node attributes mapping
sources = dict(zip(skills["taxonomy_skill"], skills["taxonomy_source"]))

print("4. Computing Louvain Ecosystems...")
partition = community_louvain.best_partition(G, weight='weight')
nx.set_node_attributes(G, partition, 'group')

print("5. Computing Network Centralities (Gatekeepers & Dominance)...")
print("   - Betweenness Centrality (Gatekeepers)...")
betweenness = nx.betweenness_centrality(G, weight='distance')
nx.set_node_attributes(G, betweenness, 'betweenness')

print("   - Eigenvector Centrality (Structural Dominance)...")
try:
    eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=2000)
except:
    print("   Eigenvector failed to converge, falling back to degree centrality")
    eigenvector = nx.degree_centrality(G)
nx.set_node_attributes(G, eigenvector, 'eigenvector')

# Apply metadata
for node in G.nodes():
    G.nodes[node]['freq'] = int(skill_counts.get(node, 1))
    G.nodes[node]['category'] = categorize_skill(node)
    G.nodes[node]['source'] = sources.get(node, "unknown")
    G.nodes[node]['label'] = node

print("6. Saving wgi_graph.pkl...")
with open("wgi_graph.pkl", "wb") as f:
    pickle.dump(G, f)

print("✅ Success!")
