"""IMAP email fetching and attachment extraction."""

import email
import imaplib
from email.message import Message

from loguru import logger

from AiRecAgent.settings import Settings

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


def _is_resume_attachment(part: Message) -> bool:
    """Return True if this MIME part looks like a resume file."""
    disposition = part.get_content_disposition() or ""
    filename = part.get_filename() or ""
    if "attachment" not in disposition and not filename:
        return False
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in SUPPORTED_EXTENSIONS


def fetch_resume_attachments(settings: Settings) -> list[tuple[str, bytes]]:
    """Connect via IMAP, fetch unseen emails and return (filename, bytes) pairs.

    Returns an empty list if IMAP is disabled or credentials are missing.
    """
    if not settings.imap_enabled or not settings.imap_user:
        logger.info("IMAP not configured, skipping email fetch")
        return []

    results: list[tuple[str, bytes]] = []
    try:
        conn = imaplib.IMAP4_SSL(settings.imap_host, settings.imap_port)
        conn.login(settings.imap_user, settings.imap_pass)
        conn.select("INBOX")

        _, message_ids = conn.search(None, "UNSEEN")
        ids = (message_ids[0] or b"").split()
        logger.info("Found {} unseen emails", len(ids))

        for msg_id in ids:
            _, msg_data = conn.fetch(msg_id.decode(), "(RFC822)")
            if not msg_data or not msg_data[0]:
                continue
            raw = msg_data[0][1] if isinstance(msg_data[0], tuple) else None
            if not isinstance(raw, bytes):
                continue

            msg = email.message_from_bytes(raw)
            for part in msg.walk():
                if _is_resume_attachment(part):
                    filename = part.get_filename() or f"resume_{msg_id.decode()}.txt"
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        results.append((filename, payload))
                        logger.info("Fetched attachment: {}", filename)

        conn.logout()
    except Exception as exc:
        logger.error("IMAP fetch failed: {}", exc)

    return results
