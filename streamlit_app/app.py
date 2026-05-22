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
        st.error(f"Ошибка API: {exc}")
        return None


def _post_json(path: str, body: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Ошибка API: {exc}")
        return None


def _delete(path: str) -> bool:
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        st.error(f"Ошибка API: {exc}")
        return False


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
        st.error(f"Ошибка API: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar — Jobs
# ---------------------------------------------------------------------------

st.sidebar.header("Вакансии")

with st.sidebar.expander("➕ Создать вакансию"):
    with st.form("create_job_form"):
        title = st.text_input("Название")
        description = st.text_area("Описание", height=120)
        requirements = st.text_area("Требования (необязательно)", height=80)
        if st.form_submit_button("Создать"):
            result = _post_json(
                "/jobs",
                {
                    "title": title,
                    "description": description,
                    "requirements": requirements or None,
                },
            )
            if result:
                st.success(f"Вакансия #{result['id']} создана")
                st.rerun()

jobs_data = _get("/jobs") or []
jobs_list = (
    sorted(jobs_data, key=lambda j: j["id"]) if isinstance(jobs_data, list) else []
)
jobs = {j["id"]: j for j in jobs_list}

if not jobs:
    st.sidebar.info("Вакансий пока нет. Создайте выше.")
    selected_job_id = None
else:
    job_ids = [j["id"] for j in jobs_list]
    selected_job_id = st.sidebar.selectbox(
        "Выберите вакансию",
        options=job_ids,
        format_func=lambda x: f"#{x} {jobs[x]['title']}",
    )

st.sidebar.divider()

# Upload resume
st.sidebar.header("Загрузить резюме")
uploaded = st.sidebar.file_uploader(
    "PDF, DOCX или TXT",
    type=["pdf", "docx", "doc", "txt"],
)
if uploaded and st.sidebar.button("Загрузить"):
    result = _post_file("/resumes/upload", uploaded.read(), uploaded.name)
    if result:
        st.sidebar.success(
            f"Импортировано: {result.get('name') or result.get('email') or 'Неизвестно'}"
        )

st.sidebar.divider()

# Email inbox stats — auto-refreshes every 30 s to match the background poll interval
st.sidebar.header("Входящие письма")


@st.fragment(run_every=30)
def _email_stats_widget() -> None:
    result = _get("/email/stats")
    if result:
        col1, col2 = st.columns(2)
        col1.metric("Проверено писем", result["emails_checked"])
        col2.metric("Найдено вложений", result["attachments_found"])
        last = result.get("last_poll_at")
        if last:
            st.caption(f"Последняя проверка: {last[:19].replace('T', ' ')} UTC")
        else:
            st.caption("Ожидание первой проверки…")


with st.sidebar:
    _email_stats_widget()

# ---------------------------------------------------------------------------
# Main panel — Recommendations
# ---------------------------------------------------------------------------


def _render_recs(recs: list, top_k: int) -> None:
    """Render a ranked candidate list."""
    if not recs:
        st.info("Кандидаты не найдены. Сначала загрузите резюме.")
        return
    st.markdown(f"### Топ {len(recs)} кандидатов")
    for rank, rec in enumerate(recs, start=1):
        c = rec["candidate"]
        name = c.get("name") or c.get("email") or f"Кандидат #{c['id']}"
        overall = rec["overall_score"]
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**#{rank} — {name}**")
                if c.get("email"):
                    st.caption(f"✉ {c['email']}")
                if c.get("skills"):
                    st.markdown("**Навыки:** " + ", ".join(c["skills"][:10]))
                if c.get("experience_years") is not None:
                    st.caption(f"Опыт: {c['experience_years']:.1f} лет")
                if c.get("education"):
                    st.caption(f"Образование: {c['education']}")
            with col2:
                st.metric("Итого", f"{overall:.2%}")
                scores = {
                    "Семантика": rec.get("semantic_score"),
                    "TF-IDF": rec.get("tfidf_score"),
                    "LLM": rec.get("llm_score"),
                }
                for label, val in scores.items():
                    if val is not None:
                        st.caption(f"{label}: {val:.2%}")
            if rec.get("explanation"):
                st.info(f"💬 {rec['explanation']}")


tab_stored, tab_adhoc = st.tabs(["По вакансии из базы", "Новая вакансия (текст)"])

with tab_stored:
    if selected_job_id is None:
        st.info(
            "Выберите или создайте вакансию на боковой панели, чтобы увидеть рекомендации."
        )
    else:
        job = jobs[selected_job_id]
        st.subheader(f"Вакансия: {job['title']}")
        with st.expander("Описание вакансии"):
            st.write(job["description"])
            if job.get("requirements"):
                st.markdown("**Требования:**")
                st.write(job["requirements"])

        top_k = st.slider(
            "Топ кандидатов", min_value=1, max_value=20, value=5, key="top_k_stored"
        )

        with st.spinner("Оценка кандидатов…"):
            data = _get(
                "/recommendations", params={"job_id": selected_job_id, "limit": top_k}
            )

        if data is None or not isinstance(data, dict):
            st.warning("Данные не получены.")
        else:
            _render_recs(data.get("recommendations", []), top_k)

with tab_adhoc:
    st.markdown(
        "Вставьте текст вакансии, чтобы немедленно получить подборку кандидатов — без сохранения в базе."
    )
    adhoc_text = st.text_area(
        "Текст вакансии",
        height=200,
        key="adhoc_job_text",
        placeholder="Например: Требуется Python-разработчик с опытом FastAPI и PostgreSQL…",
    )
    top_k_adhoc = st.slider(
        "Топ кандидатов", min_value=1, max_value=20, value=5, key="top_k_adhoc"
    )

    if st.button("Найти кандидатов", key="adhoc_search"):
        if not adhoc_text.strip():
            st.warning("Введите текст вакансии.")
        else:
            with st.spinner("Оценка кандидатов…"):
                data = _get(
                    "/recommendations",
                    params={"job_text": adhoc_text, "limit": top_k_adhoc},
                )
            if data is None or not isinstance(data, dict):
                st.warning("Данные не получены.")
            else:
                _render_recs(data.get("recommendations", []), top_k_adhoc)

# ---------------------------------------------------------------------------
# Candidates table
# ---------------------------------------------------------------------------

with st.expander("Все кандидаты"):
    candidates_data = _get("/candidates") or []
    if isinstance(candidates_data, list) and candidates_data:
        header = st.columns([1, 2, 2, 3, 1, 2, 2, 1])
        for col, label in zip(
            header,
            ["ID", "Имя", "Email", "Навыки", "Опыт", "Образование", "Файл", ""],
        ):
            col.markdown(f"**{label}**")
        for c in candidates_data:
            cols = st.columns([1, 2, 2, 3, 1, 2, 2, 1])
            cols[0].write(c["id"])
            cols[1].write(c.get("name") or "—")
            cols[2].write(c.get("email") or "—")
            cols[3].write(", ".join((c.get("skills") or [])[:5]) or "—")
            cols[4].write(
                f"{c['experience_years']:.1f}"
                if c.get("experience_years") is not None
                else "—"
            )
            cols[5].write(c.get("education") or "—")
            cols[6].write(c.get("source_file") or "—")
            if cols[7].button("🗑", key=f"del_{c['id']}"):
                if _delete(f"/candidates/{c['id']}"):
                    st.rerun()
    else:
        st.info("Кандидатов пока нет.")
