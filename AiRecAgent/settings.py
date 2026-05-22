import enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from yarl import URL

# <project_root>/AiRecAgent/settings.py → .parent.parent = project root
_PROJECT_ROOT = Path(__file__).parent.parent
RESUME_UPLOAD_DIR = _PROJECT_ROOT / "resumes"


class LogLevel(enum.StrEnum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO
    # Variables for the database
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "AiRecAgent"
    db_pass: str = "AiRecAgent"  # noqa: S105
    db_base: str = "AiRecAgent"
    db_echo: bool = False

    # IMAP settings for fetching resumes from email
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    imap_user: str = ""
    imap_pass: str = ""
    imap_enabled: bool = False

    # LLM
    anthropic_api_key: str = ""

    # Embedding model (multilingual, handles RU + EN)
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Resume file storage
    resume_upload_dir: Path = RESUME_UPLOAD_DIR

    @property
    def db_url(self) -> URL:
        """
        Assemble database URL from settings.

        :return: database URL.
        """
        return URL.build(
            scheme="postgresql+asyncpg",
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_pass,
            path=f"/{self.db_base}",
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AIRECAGENT_",
        env_file_encoding="utf-8",
    )


settings = Settings()
