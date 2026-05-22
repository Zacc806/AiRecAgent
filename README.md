# AI Recruiting Agent

ИИ-ассистент для рекрутинга, который автоматически получает резюме из электронной почты, разбирает их и ранжирует кандидатов по описаниям вакансий с помощью ансамбля из трёх моделей (эмбеддинги Sentence-BERT + TF-IDF + LLM).

---

## Содержание

1. [Архитектура](#архитектура)
2. [Конвейер обработки](#конвейер-обработки)
3. [Модели ранжирования](#модели-ранжирования)
4. [API: справочник и примеры](#api-справочник-и-примеры)
5. [Streamlit UI](#streamlit-ui)
6. [Настройка и конфигурация](#настройка-и-конфигурация)
7. [Запуск проекта](#запуск-проекта)
8. [Запуск тестов](#запуск-тестов)

---

## Архитектура

```
┌──────────────────────────────────────────────────────────────┐
│                       Загрузка данных                        │
│                                                              │
│  Email (IMAP)  ──┐                                           │
│                  ├──► ResumeParser ──► CandidateModel (БД)   │
│  Загрузка файла──┘                                           │
│                                                              │
│  Ручной ввод   ──────► JobParser   ──► JobModel (БД)         │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    NLP-конвейер (spaCy)                      │
│                                                              │
│  Токенизация → Распознавание именованных сущностей → Ранжирование ключевых слов │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   Оркестратор ранжирования                   │
│                                                              │
│  ┌─────────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │ Sentence-BERT   │  │  TF-IDF     │  │  LLM (Claude)    │  │
│  │  (вес 40%)      │  │ (вес 30%)   │  │  (вес 30%)       │  │
│  └────────┬────────┘  └──────┬──────┘  └────────┬─────────┘  │
│           └──────────────────┴──────────────────┘            │
│                              │                               │
│                   Взвешенная средняя оценка                  │
│                   + Кэш в MatchModel                         │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                     REST API (FastAPI)                       │
│              GET /api/v1/recommendations?job_id=1            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│         Управление вакансиями · Загрузка резюме · Рейтинги   │
└──────────────────────────────────────────────────────────────┘
```

**Технологический стек:**

| Слой | Технология |
|---|---|
| Backend API | FastAPI + Uvicorn (async) |
| База данных | PostgreSQL 18 через async SQLAlchemy |
| Эмбеддинги | `sentence-transformers` (многоязычный MiniLM) |
| TF-IDF | `scikit-learn` |
| LLM оценка | Anthropic Claude (`claude-haiku-4-5-20251001`) |
| NLP | spaCy |
| Парсинг файлов | pypdf, python-docx |
| Фронтенд | Streamlit |
| Контейнеризация | Docker + Docker Compose |

---

## Конвейер обработки

### Этап 1 — Получение писем

Фоновый воркер (`services/email_service.py`) опрашивает настроенный IMAP-ящик каждые 30 секунд. Использует метку UID, чтобы не загружать уже обработанные письма повторно.

```
Запуск / перезапуск
      │
      ▼
Записать UIDNEXT (метку) из входящих
      │
      └─► Каждые 30 с:
              │
              ├─ IMAP FETCH UIDs > метка
              │
              ├─ Для каждого письма с вложением PDF/DOCX/TXT:
              │     ├─ Скачать байты вложения
              │     ├─ Сохранить в resumes/ с префиксом UUID
              │     └─ Передать в ResumeParser
              │
              └─ Обновить статистику (emails_checked, attachments_found, last_poll_at)
```

Поддерживаемые типы вложений: `.pdf`, `.docx`, `.doc`, `.txt`

Та же логика загрузки запускается при ручном вызове `POST /api/v1/email/poll`.

---

### Этап 2 — Парсинг резюме

`services/resume_parser.py` извлекает структурированные данные из файлов резюме.

**Извлечение текста:**
- PDF → постраничный текст через `pypdf`
- DOCX → текст абзацев через `python-docx`
- TXT → декодирование UTF-8

**Структурированное извлечение (два режима):**

| Режим | Условие | Извлекает |
|---|---|---|
| **LLM** | Установлен `ANTHROPIC_API_KEY` | имя, email, навыки (нормализованы на русском), годы опыта, образование (по-русски), полные NLP-данные |
| **Regex + NLP (резервный)** | Нет ключа API | email через regex, имя через сущность spaCy PERSON, навыки через сопоставление ключевых слов, опыт через regex подсчёта лет |

Поля, сохраняемые в `CandidateModel`:

```
name, email, raw_text, skills (JSON-массив), experience_years (float),
education (str), embedding (вектор Sentence-BERT), nlp_data (JSON),
source_file
```

---

### Этап 3 — Парсинг вакансий

`services/job_parser.py` обрабатывает описания вакансий, переданных через `POST /api/v1/jobs`.

**Два режима (та же логика условий, что и у парсера резюме):**
- LLM: извлекает `required_skills`, `experience_level`, `tech_stack`
- Regex: эвристика уровня опыта по текстовым паттернам

В обоих режимах всегда запускается NLP-конвейер для заполнения `nlp_keywords`.

---

### Этап 4 — NLP-конвейер

`services/nlp_pipeline.py` запускает три стадии spaCy на любом тексте:

| Стадия | Результат |
|---|---|
| Токенизация | Словесные токены + границы предложений (пробелы исключены) |
| NER | Именованные сущности, сгруппированные по типу: PERSON, ORG, DATE, GPE, … |
| Ранжирование ключевых слов | Топ-токены по частоте термина (не стоп-слова, алфавитные, ≥3 символа) |

Результат сохраняется как JSON в `candidates.nlp_data` и `jobs.nlp_keywords`.

---

### Этап 5 — Ранжирование и оценка

`services/matching/orchestrator.py` координирует все три модели и объединяет результаты с кэшем `MatchModel`.

**Поток оценки для `GET /recommendations?job_id=X`:**

```
1. Загрузить JobModel из БД
2. Сформировать job_text = title + description + requirements + tech_stack + required_skills
3. Семантика: закодировать job_text → эмбеддинг 384 измерения (кэш в jobs.embedding)
4. TF-IDF: обучить векторизатор на (все raw_texts кандидатов + job_text), оценить каждого
5. LLM:    для некэшированных кандидатов вызвать Claude с job_text + фрагментом резюме
6. Вычислить overall_score = 0.4 × semantic + 0.3 × tfidf + 0.3 × llm
7. Upsert строк MatchModel (кэшировать оценки + объяснение)
8. Вернуть top-K, отсортированных по overall_score
```

**Инвалидация кэша:** передайте `?refresh=true` для принудительного пересчёта всех кандидатов.

---

## Модели ранжирования

### 1. Семантическое сходство (Sentence-BERT)

**Модель:** `paraphrase-multilingual-MiniLM-L12-v2`
**Поддержка:** русский + английский (нативно многоязычная)

Каждый текст кодируется в 384-мерный L2-нормализованный вектор. Сходство вычисляется как косинусное сходство, затем переводится из [-1, 1] в [0, 1]:

```
score = (cosine_similarity + 1) / 2
```

Эмбеддинги кандидатов и вакансий хранятся в базе данных и переиспользуются между запросами. Модель улавливает семантические связи даже при отсутствии точных совпадений ключевых слов (например, «разработка ПО» ≈ «software development»).

---

### 2. TF-IDF + косинусное сходство

**Реализация:** `scikit-learn` TfidfVectorizer

Конфигурация:
- Анализатор: по словам, унограммы + биграммы (`ngram_range=(1, 2)`)
- `sublinear_tf=True` (логарифмическое масштабирование частоты термина)
- `strip_accents="unicode"` (нормализует символы с диакритикой)
- Обучается на полном корпусе всех текстов кандидатов + текст вакансии

Компонент IDF поощряет термины, специфичные для конкретного резюме, а не общие для всех кандидатов. Этот базовый метод быстрый, детерминированный и не требует ключа API.

---

### 3. LLM оценка (Claude)

**Модель:** `claude-haiku-4-5-20251001`

LLM получает структурированный промпт с описанием вакансии и усечённым фрагментом резюме (≤2000 символов). Возвращает JSON-объект:

```json
{ "score": 0.82, "explanation": "Кандидат имеет 4 года опыта в Python и Django, что соответствует требованиям вакансии. Опыт работы с PostgreSQL и Redis совпадает с техническим стеком. Опыт с ML-инструментами является преимуществом." }
```

Оценка находится в диапазоне [0, 1], объяснение — на русском языке. Оценки LLM кэшируются в `MatchModel` и переиспользуются, если не передан параметр `?refresh=true`.

Если `ANTHROPIC_API_KEY` не установлен, компонент LLM пропускается, а итоговая оценка перераспределяется между семантической моделью и TF-IDF.

---

### Весовые коэффициенты

| Модель | Вес | Обоснование |
|---|---|---|
| Семантическая (Sentence-BERT) | 40% | Лучше всего справляется с межъязыковым и парафразным сопоставлением |
| LLM (Claude) | 30% | Целостная контекстуальная оценка |
| TF-IDF | 30% | Быстрый, точный по ключевым словам, без зависимости от API |

---

## API: справочник и примеры

**Базовый URL:** `http://localhost:8000/api/v1`
**Интерактивная документация:** `http://localhost:8000/api/docs`

---

### Вакансии

#### Создать вакансию

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Senior Python Developer",
    "description": "We are looking for a backend engineer to build scalable APIs.",
    "requirements": "5+ years Python, FastAPI or Django, PostgreSQL, Docker, CI/CD experience"
  }'
```

Ответ `201 Created`:
```json
{
  "id": 1,
  "title": "Senior Python Developer",
  "description": "We are looking for a backend engineer to build scalable APIs.",
  "requirements": "5+ years Python, FastAPI or Django, PostgreSQL, Docker, CI/CD experience",
  "required_skills": ["Python", "FastAPI", "Django", "PostgreSQL", "Docker"],
  "experience_level": "senior",
  "tech_stack": ["Python", "FastAPI", "PostgreSQL", "Docker"],
  "nlp_keywords": ["python", "backend", "scalable", "apis", "engineer"],
  "created_at": "2026-05-22T10:00:00"
}
```

#### Список вакансий

```bash
curl "http://localhost:8000/api/v1/jobs?limit=10&offset=0"
```

#### Получить одну вакансию

```bash
curl http://localhost:8000/api/v1/jobs/1
```

---

### Кандидаты / Резюме

#### Загрузить файл резюме

```bash
curl -X POST http://localhost:8000/api/v1/resumes/upload \
  -F "file=@/path/to/resume.pdf"
```

Ответ `201 Created`:
```json
{
  "id": 42,
  "email": "ivan@example.com",
  "name": "Иван Петров",
  "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "Redis"],
  "experience_years": 6.0,
  "education": "Московский государственный университет, Computer Science",
  "source_file": "abc123_resume.pdf",
  "created_at": "2026-05-22T10:05:00"
}
```

#### Список всех кандидатов

```bash
curl "http://localhost:8000/api/v1/candidates?limit=50&offset=0"
```

#### Удалить кандидата

```bash
curl -X DELETE http://localhost:8000/api/v1/candidates/42
```

Возвращает `204 No Content`. Каскадно удаляет все кэшированные оценки совпадений для этого кандидата.

---

### Рекомендации

#### По сохранённому ID вакансии (использует кэш)

```bash
curl "http://localhost:8000/api/v1/recommendations?job_id=1&limit=5"
```

Ответ:
```json
{
  "job_id": 1,
  "recommendations": [
    {
      "candidate": {
        "id": 42,
        "name": "Иван Петров",
        "email": "ivan@example.com",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "Redis"],
        "experience_years": 6.0,
        "education": "МГУ, Computer Science",
        "source_file": "abc123_resume.pdf",
        "created_at": "2026-05-22T10:05:00"
      },
      "semantic_score": 0.87,
      "tfidf_score": 0.79,
      "llm_score": 0.85,
      "overall_score": 0.84,
      "explanation": "Кандидат имеет 6 лет опыта Python и опыт работы с FastAPI, PostgreSQL и Docker, что полностью соответствует требованиям. Навыки работы с Redis являются дополнительным преимуществом."
    },
    {
      "candidate": { "id": 17, "name": "Алексей Смирнов", "..." : "..." },
      "semantic_score": 0.74,
      "tfidf_score": 0.68,
      "llm_score": 0.71,
      "overall_score": 0.71,
      "explanation": "Кандидат соответствует большинству требований, однако опыт с FastAPI ограничен."
    }
  ]
}
```

#### Принудительный пересчёт всех кандидатов

```bash
curl "http://localhost:8000/api/v1/recommendations?job_id=1&limit=5&refresh=true"
```

#### По произвольному тексту вакансии (без сохранённой вакансии)

```bash
curl -G "http://localhost:8000/api/v1/recommendations" \
  --data-urlencode "job_text=Senior ML Engineer, 3+ years experience with PyTorch, model deployment, REST APIs" \
  --data-urlencode "limit=3"
```

---

### Интеграция с почтой

#### Запустить ручной опрос почты

```bash
curl -X POST http://localhost:8000/api/v1/email/poll
```

Ответ:
```json
{ "fetched": 3, "imported": 3 }
```

#### Получить статистику опроса почты

```bash
curl http://localhost:8000/api/v1/email/stats
```

Ответ:
```json
{
  "last_poll_at": "2026-05-22T10:10:00",
  "emails_checked": 47,
  "attachments_found": 12
}
```

---

## Streamlit UI

Фронтенд Streamlit доступен по адресу `http://localhost:8501`.

**Боковая панель:**
- Создание и выбор вакансий
- Загрузка резюме (PDF / DOCX / TXT)
- Просмотр статистики опроса почты в реальном времени

**Главная панель — две вкладки:**
1. **«По вакансии из базы»** — выберите сохранённую вакансию и просмотрите топ-K кандидатов, отранжированных по общей оценке, с разбивкой по каждой модели и объяснением от Claude
2. **«Новая вакансия (текст)»** — вставьте любое описание вакансии для разовой оценки всех кандидатов

**Таблица кандидатов:** раскрывающийся раздел со всеми сохранёнными кандидатами и возможностью удалить отдельные записи.

---

## Настройка и конфигурация

Скопируйте `.env.example` в `.env` и заполните значения:

```bash
# Сервер
AIRECAGENT_RELOAD=True
AIRECAGENT_PORT=8000
AIRECAGENT_ENVIRONMENT=dev

# База данных (настраивается автоматически в Docker; нужна только для локального запуска)
AIRECAGENT_DB_HOST=localhost
AIRECAGENT_DB_PORT=5432
AIRECAGENT_DB_USER=AiRecAgent
AIRECAGENT_DB_PASS=AiRecAgent
AIRECAGENT_DB_BASE=AiRecAgent

# LLM (необязательно — TF-IDF + семантика работают без этого)
AIRECAGENT_ANTHROPIC_API_KEY=sk-ant-...

# Email / IMAP (необязательно — прямая загрузка файлов работает без этого)
AIRECAGENT_IMAP_ENABLED=True
AIRECAGENT_IMAP_HOST=imap.gmail.com
AIRECAGENT_IMAP_PORT=993
AIRECAGENT_IMAP_USER=cv@company.com
AIRECAGENT_IMAP_PASS=app-password-here
```

Для Gmail создайте [Пароль приложения](https://support.google.com/accounts/answer/185833) — стандартные пароли не работают с IMAP.

Все переменные окружения используют префикс `AIRECAGENT_`. Подробности см. в [документации pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

---

## Запуск проекта

### Docker (рекомендуется)

```bash
docker compose up --build
```

Запускает:
- `api` — FastAPI backend на порту 8000
- `streamlit` — Streamlit UI на порту 8501
- `db` — PostgreSQL на порту 5432 (внутренний)

Пересоберите образ после изменения `uv.lock` или `pyproject.toml`:

```bash
docker compose build
```

### Локально (uv)

```bash
uv sync --locked
uv run -m AiRecAgent
```

Для запуска приложения Streamlit необходимо, чтобы API был запущен. В отдельном терминале:

```bash
cd streamlit_app
uv run streamlit run app.py
```

---

## Запуск тестов

**В Docker:**

```bash
docker compose run --build --rm api pytest -vv .
docker compose down
```

**Локально** (требуется запущенный экземпляр PostgreSQL):

```bash
# Запустить только базу данных
docker compose up -d --wait db

# Запустить тесты
pytest -vv .
```

Конфигурация тестов находится в `pyproject.toml`. Тестовый набор использует отдельную базу данных (`AiRecAgent_test`) и откатывает каждый тест в транзакции. Переменная `ENVIRONMENT=pytest` отключает фоновые задачи, такие как опросчик почты.

---

## Структура проекта

```
AiRecAgent/
├── AiRecAgent/
│   ├── db/
│   │   ├── dao/              # Data Access Objects (CRUD)
│   │   └── models/           # ORM-модели SQLAlchemy
│   ├── services/
│   │   ├── matching/
│   │   │   ├── semantic.py   # Косинусное сходство Sentence-BERT
│   │   │   ├── tfidf.py      # TF-IDF + косинусное сходство
│   │   │   ├── llm.py        # Оценка и объяснение Claude
│   │   │   └── orchestrator.py  # Ансамбль + кэширование
│   │   ├── email_service.py  # Фоновый IMAP-воркер
│   │   ├── resume_parser.py  # PDF/DOCX/TXT → структурированный кандидат
│   │   ├── job_parser.py     # Текст вакансии → структурированная вакансия
│   │   └── nlp_pipeline.py   # Токенизация spaCy, NER, ключевые слова
│   ├── web/
│   │   └── api/recruiting/
│   │       ├── views.py      # Обработчики маршрутов FastAPI
│   │       └── schema.py     # Pydantic DTO запросов/ответов
│   └── settings.py           # Конфигурация pydantic-settings
├── streamlit_app/
│   └── app.py                # Streamlit UI
├── tests/                    # Набор тестов pytest
├── docker-compose.yml
└── pyproject.toml
```
