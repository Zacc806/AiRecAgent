"""Streamlit frontend for the AI Recruiting Agent."""

import requests
import streamlit as st

API_BASE = "http://api:8000/api/v1"

st.set_page_config(
    page_title="AI Recruiting Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Recruiting Agent")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get(path: str, params: dict | None = None) -> dict | list | None:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"API error: {exc}")
        return None


def _post_json(path: str, body: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"API error: {exc}")
        return None


def _post_file(path: str, file_bytes: bytes, filename: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}{path}",
            files={"file": (filename, file_bytes)},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"API error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar — Jobs
# ---------------------------------------------------------------------------

st.sidebar.header("Jobs")

with st.sidebar.expander("➕ Create new job"):
    with st.form("create_job_form"):
        title = st.text_input("Title")
        description = st.text_area("Description", height=120)
        requirements = st.text_area("Requirements (optional)", height=80)
        if st.form_submit_button("Create"):
            result = _post_json(
                "/jobs",
                {
                    "title": title,
                    "description": description,
                    "requirements": requirements or None,
                },
            )
            if result:
                st.success(f"Created job #{result['id']}")
                st.rerun()

jobs_data = _get("/jobs") or []
jobs = {j["id"]: j for j in jobs_data} if isinstance(jobs_data, list) else {}

if not jobs:
    st.sidebar.info("No jobs yet. Create one above.")
    selected_job_id = None
else:
    job_labels = {jid: f"#{jid} {j['title']}" for jid, j in jobs.items()}
    selected_job_id = st.sidebar.selectbox(
        "Select job",
        options=list(job_labels.keys()),
        format_func=lambda x: job_labels[x],
    )

st.sidebar.divider()

# Upload resume
st.sidebar.header("Upload Resume")
uploaded = st.sidebar.file_uploader(
    "PDF, DOCX, or TXT",
    type=["pdf", "docx", "doc", "txt"],
)
if uploaded and st.sidebar.button("Upload"):
    result = _post_file("/resumes/upload", uploaded.read(), uploaded.name)
    if result:
        st.sidebar.success(
            f"Imported: {result.get('name') or result.get('email') or 'Unknown'}"
        )

st.sidebar.divider()

# Email poll
st.sidebar.header("Email Inbox")
if st.sidebar.button("📬 Poll email inbox"):
    result = _post_json("/email/poll", {})
    if result:
        st.sidebar.info(
            f"Fetched {result['fetched']} emails, imported {result['imported']} resumes"
        )

# ---------------------------------------------------------------------------
# Main panel — Recommendations
# ---------------------------------------------------------------------------

if selected_job_id is None:
    st.info("Select or create a job in the sidebar to see recommendations.")
    st.stop()

job = jobs[selected_job_id]
st.subheader(f"Job: {job['title']}")
with st.expander("Job description"):
    st.write(job["description"])
    if job.get("requirements"):
        st.markdown("**Requirements:**")
        st.write(job["requirements"])

top_k = st.slider("Top candidates", min_value=1, max_value=20, value=5)

with st.spinner("Scoring candidates…"):
    data = _get("/recommendations", params={"job_id": selected_job_id, "limit": top_k})

if data is None or not isinstance(data, dict):
    st.warning("No data returned.")
    st.stop()

recs = data.get("recommendations", [])
if not recs:
    st.info("No candidates found. Upload some resumes first.")
    st.stop()

st.markdown(f"### Top {len(recs)} candidates")

for rank, rec in enumerate(recs, start=1):
    c = rec["candidate"]
    name = c.get("name") or c.get("email") or f"Candidate #{c['id']}"
    overall = rec["overall_score"]

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**#{rank} — {name}**")
            if c.get("email"):
                st.caption(f"✉ {c['email']}")
            if c.get("skills"):
                st.markdown("**Skills:** " + ", ".join(c["skills"][:10]))
            if c.get("experience_years") is not None:
                st.caption(f"Experience: {c['experience_years']:.1f} yrs")
            if c.get("education"):
                st.caption(f"Education: {c['education']}")
        with col2:
            st.metric("Overall", f"{overall:.2%}")
            scores = {
                "Semantic": rec.get("semantic_score"),
                "TF-IDF": rec.get("tfidf_score"),
                "LLM": rec.get("llm_score"),
            }
            for label, val in scores.items():
                if val is not None:
                    st.caption(f"{label}: {val:.2%}")

        if rec.get("explanation"):
            st.info(f"💬 {rec['explanation']}")

# ---------------------------------------------------------------------------
# Candidates table
# ---------------------------------------------------------------------------

with st.expander("All candidates"):
    candidates_data = _get("/candidates") or []
    if isinstance(candidates_data, list) and candidates_data:
        rows = [
            {
                "ID": c["id"],
                "Name": c.get("name") or "—",
                "Email": c.get("email") or "—",
                "Skills": ", ".join((c.get("skills") or [])[:5]),
                "Exp (yrs)": c.get("experience_years"),
                "File": c.get("source_file") or "—",
            }
            for c in candidates_data
        ]
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No candidates yet.")
