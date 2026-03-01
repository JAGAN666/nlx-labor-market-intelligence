"""
build_analytics.py
==================
Pre-computes all data needed for the NLx Labor Market Intelligence dashboard.
Reads:  colorado.csv            (raw job postings, 10,482 rows)
        colorado_processed.csv  (skill extractions, 165k rows)
Writes: analytics.pkl
"""

import pandas as pd
import numpy as np
import pickle
import re
from collections import defaultdict

# ── Colorado city lat/lon (curated from Census/Google) ────────────────────────
CITY_COORDS = {
    "Denver":           (39.7392, -104.9903),
    "Colorado Springs": (38.8339, -104.8214),
    "Aurora":           (39.7294, -104.8319),
    "Fort Collins":     (40.5853, -105.0844),
    "Boulder":          (40.0150, -105.2705),
    "Pueblo":           (38.2544, -104.6091),
    "Lakewood":         (39.7047, -105.0814),
    "Westminster":      (39.8367, -105.0372),
    "Centennial":       (39.5807, -104.8772),
    "Arvada":           (39.8028, -105.0875),
    "Highlands Ranch":  (39.5480, -104.9694),
    "Greeley":          (40.4233, -104.7091),
    "Loveland":         (40.3978, -105.0749),
    "Littleton":        (39.6136, -105.0166),
    "Thornton":         (39.8680, -104.9719),
    "Castle Rock":      (39.3722, -104.8561),
    "Broomfield":       (39.9205, -105.0867),
    "Brighton":         (39.9853, -104.8172),
    "Commerce City":    (39.8083, -104.9339),
    "Parker":           (39.5186, -104.7614),
    "Longmont":         (40.1672, -105.1019),
    "Englewood":        (39.6486, -104.9878),
    "Northglenn":       (39.8847, -104.9811),
    "Wheat Ridge":      (39.7661, -105.0772),
    "Pueblo West":      (38.3472, -104.7250),
    "Lafayette":        (39.9936, -105.0897),
    "Windsor":          (40.4772, -104.9014),
    "Federal Heights":  (39.8658, -105.0139),
    "Golden":           (39.7555, -105.2211),
    "Erie":             (40.0503, -105.0469),
    "Columbine":        (39.6286, -105.0700),
    "Ken Caryl":        (39.5872, -105.1197),
    "Montrose":         (38.4783, -107.8762),
    "Durango":          (37.2753, -107.8801),
    "Grand Junction":   (39.0639, -108.5506),
    "Pueblo":           (38.2544, -104.6091),
    "Steamboat Springs":(40.4850, -106.8317),
    "Glenwood Springs": (39.5505, -107.3248),
    "Trinidad":         (37.1694, -104.5003),
    "Alamosa":          (37.4694, -105.8700),
    "Sterling":         (40.6253, -103.2083),
    "Canon City":       (38.4408, -105.2428),
    "Florence":         (38.3908, -105.1171),
    "Johnstown":        (40.3367, -104.9158),
    "Fort Lupton":      (40.0778, -104.8033),
    "Firestone":        (40.1539, -104.9456),
    "Frederick":        (40.1039, -104.9414),
    "Brush":            (40.2594, -103.6280),
    "Weld County":      (40.6736, -104.3672),
    "Arapahoe County":  (39.6219, -104.7658),
    "Jefferson County": (39.5831, -105.2031),
}

# ── ONET major sector mapping (first 2 digits of code) ───────────────────────
ONET_SECTORS = {
    "11": "Management",
    "13": "Business & Finance",
    "15": "Computer & IT",
    "17": "Architecture & Engineering",
    "19": "Science & Research",
    "21": "Community & Social Services",
    "23": "Legal",
    "25": "Education",
    "27": "Arts & Media",
    "29": "Healthcare Practitioners",
    "31": "Healthcare Support",
    "33": "Protective Services",
    "35": "Food Service",
    "37": "Maintenance & Grounds",
    "39": "Personal Care",
    "41": "Sales",
    "43": "Office & Admin",
    "45": "Farming & Fishing",
    "47": "Construction",
    "49": "Installation & Repair",
    "51": "Production & Manufacturing",
    "53": "Transportation",
}

# Skill categories (simple keyword grouping for heatmap)
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

def categorize_skill(skill_text):
    s = str(skill_text).lower()
    for cat, keywords in SKILL_CATEGORIES.items():
        if any(kw in s for kw in keywords):
            return cat
    return "Other"

print("=== NLx Analytics Builder ===\n")

# ── 1. Load raw postings ──────────────────────────────────────────────────────
print("Loading colorado.csv …")
raw = pd.read_csv(
    "colorado.csv",
    usecols=["system_job_id","title","description","city","state","zipcode",
             "application_company","classifications_onet_code",
             "parameters_salary_min","parameters_salary_max",
             "date_compiled","ghostjob","jobclass",
             "classifications_naics_code"],
    dtype={"system_job_id": str}
)
print(f"  {len(raw):,} job postings loaded")

# Clean
raw["title"]   = raw["title"].fillna("Unknown")
raw["city"]    = raw["city"].fillna("Unknown").str.strip()
raw["company"] = raw["application_company"].fillna("Unknown").str.strip()
raw["onet"]    = raw["classifications_onet_code"].fillna("").astype(str).str[:2]
raw["sector"]  = raw["onet"].map(ONET_SECTORS).fillna("Other")
raw["is_ghost"]= raw["ghostjob"].fillna(False)
raw["salary_min"] = pd.to_numeric(raw["parameters_salary_min"], errors="coerce")
raw["salary_max"] = pd.to_numeric(raw["parameters_salary_max"], errors="coerce")
raw["description"] = raw["description"].fillna("")

# ── 2. Load processed skills ──────────────────────────────────────────────────
print("Loading colorado_processed.csv …")
skills = pd.read_csv(
    "colorado_processed.csv",
    dtype={"Research ID": str}
)
skills.columns = ["job_id","raw_skill","taxonomy_skill","taxonomy_desc","taxonomy_source","correlation"]
print(f"  {len(skills):,} skill extractions across {skills['job_id'].nunique():,} jobs")

# ── 3. Market Overview ────────────────────────────────────────────────────────
print("Computing market overview …")

city_counts = raw[~raw["is_ghost"]]["city"].value_counts().head(20).to_dict()
sector_counts = raw["sector"].value_counts().to_dict()
company_counts = raw["company"].value_counts().head(20).to_dict()
ghost_pct = round(raw["is_ghost"].mean() * 100, 1)
total_jobs = len(raw)
total_unique_jobs = int(skills["job_id"].nunique())
total_skill_extractions = len(skills)
unique_taxonomy_skills = int(skills["taxonomy_skill"].nunique())

# ONET title distribution
onet_full = raw["classifications_onet_code"].value_counts().head(20).to_dict()

overview = {
    "total_jobs": total_jobs,
    "total_with_skills": total_unique_jobs,
    "total_skill_extractions": total_skill_extractions,
    "unique_taxonomy_skills": unique_taxonomy_skills,
    "ghost_pct": ghost_pct,
    "city_counts": city_counts,
    "sector_counts": sector_counts,
    "company_counts": company_counts,
    "onet_top": onet_full,
    "taxonomy_sources": skills["taxonomy_source"].value_counts().to_dict(),
    "date_range": {"start": raw["date_compiled"].min(), "end": raw["date_compiled"].max()},
}

# ── 4. Top Skills ─────────────────────────────────────────────────────────────
print("Computing top skills …")

# Skills by taxonomy_skill (normalized)
top_skills_global = (
    skills.groupby("taxonomy_skill")
          .agg(job_count=("job_id","nunique"), mention_count=("job_id","count"),
               avg_correlation=("correlation","mean"),
               source=("taxonomy_source","first"),
               description=("taxonomy_desc","first"))
          .reset_index()
          .sort_values("job_count", ascending=False)
)
top_skills_global["category"] = top_skills_global["taxonomy_skill"].apply(categorize_skill)

# Per-city skill counts (for filter)
city_skill_counts = {}
valid_cities = raw["city"].value_counts().head(15).index.tolist()
job_city = raw.set_index("system_job_id")["city"]

skills_with_city = skills.merge(
    raw[["system_job_id","city"]].rename(columns={"system_job_id":"job_id"}),
    on="job_id", how="left"
)

for city in valid_cities:
    city_df = skills_with_city[skills_with_city["city"] == city]
    top = (
        city_df.groupby("taxonomy_skill")["job_id"]
               .nunique()
               .reset_index()
               .rename(columns={"job_id":"count"})
               .sort_values("count", ascending=False)
               .head(30)
    )
    city_skill_counts[city] = top.to_dict("records")

skills_data = {
    "global_top50": top_skills_global.head(50).to_dict("records"),
    "city_skills": city_skill_counts,
    "valid_cities": valid_cities,
}

# ── 5. Sector × Skill Heatmap ─────────────────────────────────────────────────
print("Computing sector × skill heatmap …")

skills_with_sector = skills_with_city.merge(
    raw[["system_job_id","sector"]].rename(columns={"system_job_id":"job_id"}),
    on="job_id", how="left"
)
skills_with_sector["category"] = skills_with_sector["taxonomy_skill"].apply(categorize_skill)

# For each sector, what % of jobs require each skill category
heatmap_rows = []
sectors_ordered = [s for s in ONET_SECTORS.values() if sector_counts.get(s,0) > 30]

for sector in sectors_ordered:
    sector_jobs = raw[raw["sector"]==sector]["system_job_id"].astype(str).unique()
    n_jobs = len(sector_jobs)
    if n_jobs < 5: continue
    row = {"sector": sector, "n_jobs": int(n_jobs)}
    sec_skills = skills_with_sector[skills_with_sector["sector"]==sector]
    for cat in SKILL_CATEGORIES:
        cat_jobs = sec_skills[sec_skills["category"]==cat]["job_id"].nunique()
        row[cat] = round(cat_jobs / n_jobs * 100, 1)
    heatmap_rows.append(row)

heatmap_data = {
    "rows": heatmap_rows,
    "skill_categories": list(SKILL_CATEGORIES.keys()),
}

# ── 6. Job Search Index ───────────────────────────────────────────────────────
print("Building job search index …")

# Build a searchable record per job (title + top skills)
job_skills_grouped = (
    skills.groupby("job_id")
          .apply(lambda g: g.nlargest(12, "correlation")[["raw_skill","taxonomy_skill","correlation"]].to_dict("records"))
          .reset_index(name="skills")
)

job_index_df = raw[["system_job_id","title","city","company","sector","description","salary_min","salary_max"]].copy()
job_index_df["system_job_id"] = job_index_df["system_job_id"].astype(str)
job_index_df = job_index_df.merge(
    job_skills_grouped.rename(columns={"job_id":"system_job_id"}),
    on="system_job_id", how="left"
)
job_index_df["skills"] = job_index_df["skills"].apply(lambda x: x if isinstance(x, list) else [])
job_index_df["desc_short"] = job_index_df["description"].str[:600]

# Build fast search text column
def build_search_text(row):
    skill_names = " ".join(s.get("taxonomy_skill","") for s in (row["skills"] or []))
    return f'{row["title"]} {row["company"]} {skill_names}'.lower()

job_index_df["search_text"] = job_index_df.apply(build_search_text, axis=1)
job_index = job_index_df.drop(columns=["description"]).to_dict("records")
print(f"  {len(job_index):,} job records in search index")

# ── 7. Pipeline Stats ─────────────────────────────────────────────────────────
print("Computing pipeline stats …")

corr_hist, corr_bins = np.histogram(
    skills["correlation"].dropna(),
    bins=20, range=(0, 1)
)

pipeline_data = {
    "total_raw_postings": total_jobs,
    "postings_with_skills": total_unique_jobs,
    "total_extractions": total_skill_extractions,
    "unique_raw_skills": int(skills["raw_skill"].nunique()),
    "unique_taxonomy_skills": unique_taxonomy_skills,
    "source_breakdown": skills["taxonomy_source"].value_counts().to_dict(),
    "avg_skills_per_job": round(len(skills) / total_unique_jobs, 1),
    "correlation_hist": corr_hist.tolist(),
    "correlation_bins": [round(float(b),2) for b in corr_bins.tolist()],
    "methodology": {
        "step1": "Raw job descriptions fetched from National Labor Exchange (Colorado, Jan 2026)",
        "step2": "NLP skill span extraction via SBERT (all-MiniLM-L6-v2) semantic matching",
        "step3": "Skill spans matched to ESCO / O*NET taxonomy via cosine similarity threshold ≥ 0.5",
        "step4": "Correlation coefficient = cosine similarity between raw skill embedding and taxonomy skill embedding",
    },
    "limitations": [
        "Salary data available for only 9 of 10,482 postings (<0.1%) — insufficient for salary analysis",
        "17.2% of postings flagged as 'ghost jobs' (reposted/expired) — included but labeled",
        "SBERT matching may miss domain-specific jargon not in training data",
        "Data snapshot: January 1–14, 2026 only — may not represent seasonal variation",
        "Skills extracted from text descriptions only — structured fields (ONET codes) used for sector classification",
    ],
}

# ── 8. Map Data ──────────────────────────────────────────────────────────────
print("Computing map data …")

# Per-city stats for map bubbles
city_job_counts = raw["city"].value_counts().to_dict()
city_sector_top = raw.groupby("city")["sector"].agg(lambda x: x.value_counts().index[0]).to_dict()

# Join skill info for cities
skills_city = skills.merge(
    raw[["system_job_id","city","sector"]].rename(columns={"system_job_id":"job_id"}),
    on="job_id", how="left"
)

map_cities = []
for city, (lat, lon) in CITY_COORDS.items():
    n_jobs = city_job_counts.get(city, 0)
    if n_jobs == 0:
        continue
    top_sector = city_sector_top.get(city, "Other")
    city_skills_df = skills_city[skills_city["city"]==city]
    top_skills_city = (
        city_skills_df.groupby("taxonomy_skill")["job_id"]
                      .nunique()
                      .reset_index()
                      .rename(columns={"job_id":"count"})
                      .sort_values("count", ascending=False)
                      .head(6)
    )
    # Sector breakdown for this city
    sec_counts = raw[raw["city"]==city]["sector"].value_counts().head(4).to_dict()
    map_cities.append({
        "city":      city,
        "lat":       lat,
        "lon":       lon,
        "n_jobs":    int(n_jobs),
        "top_sector": top_sector,
        "top_skills": top_skills_city.to_dict("records"),
        "sectors":   {k: int(v) for k,v in sec_counts.items()},
    })

map_cities.sort(key=lambda x: x["n_jobs"], reverse=True)
map_data = {"cities": map_cities}
print(f"  {len(map_cities)} cities mapped")

# ── 9. Save ───────────────────────────────────────────────────────────────────
print("Saving analytics.pkl …")
analytics = {
    "overview": overview,
    "skills": skills_data,
    "heatmap": heatmap_data,
    "job_index": job_index,
    "pipeline": pipeline_data,
    "map": map_data,
}

with open("analytics.pkl", "wb") as f:
    pickle.dump(analytics, f)

print("\n✅ analytics.pkl saved!")
print(f"   overview: {len(overview)} keys")
print(f"   skills:   {len(skills_data['global_top50'])} top skills")
print(f"   heatmap:  {len(heatmap_data['rows'])} sectors")
print(f"   job_index:{len(job_index)} jobs")
print(f"   pipeline: {pipeline_data['total_extractions']} extractions documented")
print(f"   map:      {len(map_data['cities'])} cities mapped")
