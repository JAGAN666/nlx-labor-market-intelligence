"""
NLx Labor Market Intelligence — Flask Backend
"""
from flask import Flask, request, jsonify, send_from_directory
import pickle
import re
import math
import networkx as nx

app = Flask(__name__, static_folder="static", static_url_path="")

print("Loading analytics…")
with open("analytics.pkl", "rb") as f:
    A = pickle.load(f)

print("Loading wgi_graph…")
try:
    with open("wgi_graph.pkl", "rb") as f:
        G = pickle.load(f)
    print(f"✅ Ready (Graph: {G.number_of_nodes()} nodes)")
except FileNotFoundError:
    G = nx.Graph()
    print("⚠️ wgi_graph.pkl not found. Graph intelligence disabled.")


def sanitize(v):
    """Convert NaN/Inf floats → None so Flask jsonify produces valid JSON null."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return v


@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/wgi")
def wgi():
    return send_from_directory("static", "wgi.html")

# ── Tab 1: Market Overview ────────────────────────────────────────────────────
@app.route("/api/overview")
def overview():
    o = A["overview"]
    return jsonify({
        "total_jobs":              o["total_jobs"],
        "total_with_skills":      o["total_with_skills"],
        "total_skill_extractions":o["total_skill_extractions"],
        "unique_taxonomy_skills": o["unique_taxonomy_skills"],
        "ghost_pct":              o["ghost_pct"],
        "city_counts":            dict(list(o["city_counts"].items())[:15]),
        "sector_counts":          o["sector_counts"],
        "company_counts":         dict(list(o["company_counts"].items())[:15]),
        "taxonomy_sources":       o["taxonomy_sources"],
        "date_range":             o["date_range"],
    })

# ── Tab 2: Top Skills ─────────────────────────────────────────────────────────
@app.route("/api/skills")
def skills():
    city  = request.args.get("city", "All")
    limit = min(int(request.args.get("limit", 30)), 50)
    s = A["skills"]
    if city == "All":
        data = s["global_top50"][:limit]
    else:
        data = s["city_skills"].get(city, s["global_top50"][:limit])[:limit]
    return jsonify({
        "skills": data,
        "cities": ["All"] + s["valid_cities"],
        "selected_city": city,
    })

# ── Tab 3: Skills × Sector Heatmap ───────────────────────────────────────────
@app.route("/api/heatmap")
def heatmap():
    return jsonify(A["heatmap"])

# ── Tab 4: Job Search ─────────────────────────────────────────────────────────
@app.route("/api/search")
def search():
    q     = request.args.get("q", "").strip().lower()
    limit = min(int(request.args.get("limit", 20)), 50)
    city  = request.args.get("city", "All")
    jobs  = A["job_index"]

    if q:
        results = [j for j in jobs if q in j.get("search_text", "")]
    else:
        results = jobs

    if city != "All":
        results = [j for j in results if j.get("city") == city]

    total = len(results)
    paged = results[:limit]

    out = []
    for j in paged:
        desc = j.get("desc_short", "") or ""
        if q and q in desc.lower():
            desc = re.compile(re.escape(q), re.IGNORECASE).sub(
                f"<mark>{q}</mark>", desc
            )
        # Sanitize skills list — remove NaN correlations
        clean_skills = []
        for s in (j.get("skills") or [])[:8]:
            clean_skills.append({
                "taxonomy_skill": s.get("taxonomy_skill", ""),
                "raw_skill":      s.get("raw_skill", ""),
                "correlation":    sanitize(s.get("correlation")),
                "taxonomy_source":s.get("taxonomy_source",""),
                "taxonomy_desc":  s.get("taxonomy_desc",""),
            })
        out.append({
            "job_id":    j.get("system_job_id", ""),
            "title":     j.get("title", ""),
            "company":   j.get("company", ""),
            "city":      j.get("city", ""),
            "sector":    j.get("sector", ""),
            "salary_min": sanitize(j.get("salary_min")),
            "salary_max": sanitize(j.get("salary_max")),
            "desc_short": desc,
            "skills":     clean_skills,
        })

    return jsonify({"results": out, "total": total, "query": q})

# ── Tab 5: Pipeline ───────────────────────────────────────────────────────────
@app.route("/api/pipeline")
def pipeline():
    return jsonify(A["pipeline"])

# ── Tab 6: Map ────────────────────────────────────────────────────────────────
@app.route("/api/map")
def map_data():
    return jsonify(A["map"])

# ══════════════════════════════════════════════════════════════════════════════
# WORKFORCE GRAPH INTELLIGENCE API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/wgi/stats")
def wgi_stats():
    groups = set(nx.get_node_attributes(G, "group").values())
    return jsonify({
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "communities": len(groups)
    })

@app.route("/api/wgi/nodes")
def wgi_nodes():
    limit = min(int(request.args.get("limit", 200)), 800)
    sort_by = request.args.get("sort_by", "freq") # freq, betweenness, eigenvector
    
    nodes = []
    for n, d in G.nodes(data=True):
        nodes.append({
            "id": n,
            "label": d.get("label", n),
            "freq": d.get("freq", 1),
            "group": d.get("group", 0),
            "betweenness": sanitize(d.get("betweenness", 0)),
            "eigenvector": sanitize(d.get("eigenvector", 0)),
            "category": d.get("category", "Other")
        })
    
    nodes.sort(key=lambda x: x[sort_by] or 0, reverse=True)
    return jsonify(nodes[:limit])

@app.route("/api/wgi/graph")
def wgi_graph():
    limit = min(int(request.args.get("limit", 200)), 800)
    
    # Get top nodes
    sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1].get("freq", 1), reverse=True)[:limit]
    top_node_ids = {n[0] for n in sorted_nodes}
    
    out_nodes = []
    for n, d in sorted_nodes:
        out_nodes.append({
            "id": n,
            "name": d.get("label", n),
            "freq": d.get("freq", 1),
            "group": d.get("group", 0)
        })
        
    out_links = []
    for u, v, d in G.edges(data=True):
        if u in top_node_ids and v in top_node_ids:
            # only send highly correlated links to keep the graph sparse and beautiful
            w = d.get("weight", 0)
            out_links.append({
                "source": u,
                "target": v,
                "weight": sanitize(w)
            })
            
    return jsonify({"nodes": out_nodes, "links": out_links})

@app.route("/api/wgi/path")
def wgi_path():
    source = request.args.get("source")
    target = request.args.get("target")

    if not source or not target:
        return jsonify({"error": "Missing source or target"}), 400
    if source not in G or target not in G:
        return jsonify({"error": "Skill not found in graph vocabulary"}), 404

    try:
        # We find the shortest path based on inverse co-occurrence (distance)
        path = nx.shortest_path(G, source=source, target=target, weight='distance')
        
        # Build node details for the path
        path_nodes = []
        for n in path:
            d = G.nodes[n]
            path_nodes.append({
                "id": n,
                "label": d.get("label", n),
                "category": d.get("category", "Other"),
                "freq": d.get("freq", 1)
            })
            
        return jsonify({"path": path_nodes, "length": len(path)-1})
    except nx.NetworkXNoPath:
        return jsonify({"error": "No career transition path found between these skills."}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5050)
