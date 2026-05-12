"""
Resume Matching Engine — Backend
Redrob AI Campus Hackathon

Run:  python3 resume_matcher_backend.py
Output: results.json  +  printed final answer
"""

import math
import json

# ─────────────────────────────────────────────────────────────
# SKILL ALIASES  (exact as provided — do not modify)
# ─────────────────────────────────────────────────────────────
SKILL_ALIASES = {
    # Languages
    "python": "python", "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    # ML / Data
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    # Web — Frontend
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css",
    "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    # Web — Backend
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    # Databases
    "sql": "sql",
    "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws",
    # Mobile
    "android": "android", "firebase": "firebase",
    # CS Fundamentals
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    # Design
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
RESUMES = [
    ("Arjun Sharma",    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair",      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta",     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel",     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh",    "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta",     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao",     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar",    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer",      "python, R, statistics, ML, regression, clustering, Power-BI"),
]

JDS = [
    {
        "id": "JD-1", "company": "Kakao", "city": "Seoul", "role": "ML Engineer",
        "required": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization",
        "preferred": "NLP, BERT, Feature Engineering, Statistics",
    },
    {
        "id": "JD-2", "company": "Naver", "city": "Seongnam", "role": "Backend Engineer",
        "required": "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes",
        "preferred": "REST API, CI/CD, Redis",
    },
    {
        "id": "JD-3", "company": "Line", "city": "Seoul", "role": "Frontend Engineer",
        "required": "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS",
        "preferred": "Node.js, GraphQL, Redux, Jest, AWS",
    },
]

# ─────────────────────────────────────────────────────────────
# STEP 1 — NORMALIZE SKILLS
# ─────────────────────────────────────────────────────────────
def normalize_skills(raw: str) -> list:
    """Split on commas, lowercase, apply alias map (multi-word first), discard unknowns."""
    tokens = [t.strip().lower() for t in raw.split(",")]
    # Sort keys longest-first so multi-word phrases match before single tokens
    sorted_keys = sorted(SKILL_ALIASES.keys(), key=len, reverse=True)
    result = []
    for token in tokens:
        for key in sorted_keys:
            if token == key:
                result.append(SKILL_ALIASES[key])
                break
    return result


# ─────────────────────────────────────────────────────────────
# STEP 2 — DEDUPLICATE
# ─────────────────────────────────────────────────────────────
def deduplicate(skills: list) -> list:
    seen = set()
    out = []
    for s in skills:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ─────────────────────────────────────────────────────────────
# STEP 3 — BUILD VOCABULARY
# ─────────────────────────────────────────────────────────────
def build_vocabulary(processed_resumes: list) -> list:
    """Sorted alphabetically from all deduplicated resume skills."""
    all_skills = set()
    for _, skills in processed_resumes:
        all_skills.update(skills)
    return sorted(all_skills)


# ─────────────────────────────────────────────────────────────
# STEP 4 — TF-IDF VECTORS
# ─────────────────────────────────────────────────────────────
def compute_tfidf_vectors(processed_resumes: list, vocab: list) -> list:
    """
    TF  = 1 / N  (after dedup each skill appears once)
    IDF = ln(10 / df)
    TF-IDF = TF * IDF
    """
    N_docs = len(processed_resumes)

    # Document frequency
    df = {skill: 0 for skill in vocab}
    for _, skills in processed_resumes:
        for s in skills:
            df[s] += 1

    # IDF
    idf = {skill: math.log(N_docs / df[skill]) for skill in vocab}

    # Build vectors
    vectors = []
    for name, skills in processed_resumes:
        N = len(skills)
        vec = []
        for skill in vocab:
            if skill in skills:
                tf = 1.0 / N
                vec.append(tf * idf[skill])
            else:
                vec.append(0.0)
        vectors.append({"name": name, "skills": skills, "N": N, "vector": vec})

    return vectors, idf, df


# ─────────────────────────────────────────────────────────────
# STEP 5 — JD BINARY VECTORS
# ─────────────────────────────────────────────────────────────
def build_jd_vectors(jds: list, vocab: list) -> list:
    """Binary 1/0 over the same vocabulary."""
    results = []
    for jd in jds:
        raw = jd["required"] + ", " + jd["preferred"]
        jd_skills = set(normalize_skills(raw))
        vec = [1.0 if skill in jd_skills else 0.0 for skill in vocab]
        results.append({**jd, "skills_normalized": sorted(jd_skills), "vector": vec})
    return results


# ─────────────────────────────────────────────────────────────
# STEP 6 — COSINE SIMILARITY + RANKING
# ─────────────────────────────────────────────────────────────
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def euclidean_norm(v):
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity(a, b):
    na, nb = euclidean_norm(a), euclidean_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)

def rank_candidates(resume_vectors, jd_vectors):
    rankings = []
    for jd in jd_vectors:
        scores = []
        for r in resume_vectors:
            sim = cosine_similarity(r["vector"], jd["vector"])
            scores.append({"name": r["name"], "score": sim})
        # Sort: descending score, alphabetical name for ties
        scores.sort(key=lambda x: (-x["score"], x["name"]))
        rankings.append({**jd, "ranking": scores})
    return rankings


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RESUME MATCHING ENGINE — Redrob AI Campus Hackathon")
    print("=" * 60)

    # Step 1 & 2
    print("\n[STEP 1 & 2] Normalize + Deduplicate")
    processed = [(name, deduplicate(normalize_skills(raw))) for name, raw in RESUMES]
    for name, skills in processed:
        print(f"  {name:20s} → {skills}")

    # Step 3
    vocab = build_vocabulary(processed)
    print(f"\n[STEP 3] Vocabulary: {len(vocab)} terms")
    print(f"  {vocab}")

    # Step 4
    resume_vectors, idf, df = compute_tfidf_vectors(processed, vocab)
    print("\n[STEP 4] IDF values")
    for skill in vocab:
        print(f"  {skill:30s} df={df[skill]}  IDF={idf[skill]:.6f}")

    print("\n[STEP 4] TF-IDF vectors (non-zero entries)")
    for r in resume_vectors:
        nonzero = [(vocab[i], round(v, 6)) for i, v in enumerate(r["vector"]) if v > 0]
        print(f"  {r['name']:20s} N={r['N']}  {nonzero}")

    # Step 5
    jd_vectors = build_jd_vectors(JDS, vocab)
    print("\n[STEP 5] JD binary vectors")
    for jd in jd_vectors:
        print(f"  {jd['id']} {jd['role']:20s} → {jd['skills_normalized']}")

    # Step 6
    rankings = rank_candidates(resume_vectors, jd_vectors)
    print("\n[STEP 6] All cosine scores")
    for jd in rankings:
        print(f"\n  {jd['id']} — {jd['company']} ({jd['role']})")
        for r in jd["ranking"]:
            print(f"    {r['name']:20s} {r['score']:.6f}")

    # Final output
    print("\n" + "=" * 60)
    print("  FINAL OUTPUT — TOP 3 CANDIDATES PER JD")
    print("=" * 60)
    final = {}
    for jd in rankings:
        top3 = jd["ranking"][:3]
        line = ", ".join(f"{r['name']}({r['score']:.2f})" for r in top3)
        label = f"{jd['id']} — {jd['company']} ({jd['role']})"
        print(f"\n{label}")
        print(line)
        final[jd["id"]] = {
            "company": jd["company"],
            "role": jd["role"],
            "top3": [{"name": r["name"], "score": round(r["score"], 6)} for r in top3],
            "all_scores": [{"name": r["name"], "score": round(r["score"], 6)} for r in jd["ranking"]],
        }

    # Export JSON for frontend
    output = {
        "vocab": vocab,
        "idf": {k: round(v, 6) for k, v in idf.items()},
        "df": df,
        "resume_skills": {name: skills for name, skills in processed},
        "jd_skills": {jd["id"]: jd["skills_normalized"] for jd in jd_vectors},
        "results": final,
    }
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n[INFO] results.json written — load this in the frontend.")
    print("=" * 60)

    return output


if __name__ == "__main__":
    main()
