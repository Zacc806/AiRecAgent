"""IMAP email fetching and attachment extraction."""

import email
import imaplib
import re
from datetime import UTC, datetime
from email.message import Message
from typing import Any

from loguru import logger

from AiRecAgent.settings import Settings

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


class _WatermarkState:
    uid: int | None = None


_watermark = _WatermarkState()

# Cumulative stats updated after every fetch_resume_attachments() call.
_stats: dict[str, Any] = {
    "last_poll_at": None,  # datetime | None
    "emails_checked": 0,  # total new emails seen across all polls
    "attachments_found": 0,  # total resume attachments extracted
}


def get_stats() -> dict[str, Any]:
    """Return a snapshot of cumulative email polling statistics."""
    return dict(_stats)


def initialize_watermark(settings: Settings) -> None:
    """Record UIDNEXT so only emails arriving after this call are ever fetched."""
    if not settings.imap_enabled or not settings.imap_user:
        return
    try:
        conn = imaplib.IMAP4_SSL(settings.imap_host, settings.imap_port)
        conn.login(settings.imap_user, settings.imap_pass)
        status, data = conn.status("INBOX", "(UIDNEXT)")
        if status == "OK" and data and data[0]:
            raw = data[0].decode() if isinstance(data[0], bytes) else str(data[0])
            match = re.search(r"UIDNEXT\s+(\d+)", raw, re.IGNORECASE)
            if match:
                _watermark.uid = int(match.group(1))
                logger.info(
                    "imap: watermark set to UID {} — prior emails ignored",
                    _watermark.uid,
                )
            else:
                logger.warning("imap: could not parse UIDNEXT from {!r}", raw)
        conn.logout()
    except Exception as exc:
        logger.error("imap: watermark init failed: {}", exc)


def _is_resume_attachment(part: Message) -> bool:
    """Return True if this MIME part looks like a resume file."""
    disposition = part.get_content_disposition() or ""
    filename = part.get_filename() or ""
    if "attachment" not in disposition and not filename:
        return False
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in SUPPORTED_EXTENSIONS


def _search_uids(conn: imaplib.IMAP4_SSL, watermark: int | None) -> list[bytes]:
    """Return UIDs to process given the current watermark."""
    if watermark is not None:
        _, uid_data = conn.uid("SEARCH", f"UID {watermark}:*")  # type: ignore[attr-defined]
        raw_uids: list[bytes] = (uid_data[0] or b"").split()
        return [u for u in raw_uids if int(u) >= watermark]
    _, message_ids = conn.search(None, "UNSEEN")
    return (message_ids[0] or b"").split()


def _fetch_uid_attachments(
    conn: imaplib.IMAP4_SSL, uid_str: str, use_uid: bool
) -> list[tuple[str, bytes]]:
    """Fetch and return resume attachments from a single message."""
    if use_uid:
        _, msg_data = conn.uid("FETCH", uid_str, "(RFC822)")  # type: ignore[attr-defined]
    else:
        _, msg_data = conn.fetch(uid_str, "(RFC822)")
    if not msg_data or not msg_data[0]:
        return []
    raw = msg_data[0][1] if isinstance(msg_data[0], tuple) else None
    if not isinstance(raw, bytes):
        return []
    results: list[tuple[str, bytes]] = []
    for part in email.message_from_bytes(raw).walk():
        if _is_resume_attachment(part):
            filename = part.get_filename() or f"resume_{uid_str}.txt"
            payload = part.get_payload(decode=True)
            if isinstance(payload, bytes):
                results.append((filename, payload))
                logger.info(
                    "imap: found attachment {!r} ({} bytes)",
                    filename,
                    len(payload),
                )
    return results


def fetch_resume_attachments(settings: Settings) -> list[tuple[str, bytes]]:
    """Fetch resume attachments from IMAP.

    When a watermark is set (via initialize_watermark), only emails arriving
    after the watermark UID are fetched. Falls back to UNSEEN if no watermark.
    Returns an empty list if IMAP is disabled or credentials are missing.
    """
    if not settings.imap_enabled or not settings.imap_user:
        return []

    results: list[tuple[str, bytes]] = []
    try:
        conn = imaplib.IMAP4_SSL(settings.imap_host, settings.imap_port)
        conn.login(settings.imap_user, settings.imap_pass)

        status, _ = conn.select("INBOX")
        if status != "OK":
            logger.error("imap: SELECT INBOX failed")
            conn.logout()
            return []

        use_uid = _watermark.uid is not None
        uids = _search_uids(conn, _watermark.uid)
        max_seen_uid: int | None = None

        for uid in uids:
            uid_str = uid.decode()
            results.extend(_fetch_uid_attachments(conn, uid_str, use_uid))
            uid_int = int(uid_str)
            if max_seen_uid is None or uid_int > max_seen_uid:
                max_seen_uid = uid_int

        if max_seen_uid is not None and _watermark.uid is not None:
            _watermark.uid = max_seen_uid + 1

        _stats["last_poll_at"] = datetime.now(UTC)
        _stats["emails_checked"] += len(uids)
        _stats["attachments_found"] += len(results)

        conn.logout()

    except imaplib.IMAP4.error as exc:
        logger.error("imap: protocol error: {}", exc)
    except OSError as exc:
        logger.error("imap: network error: {}", exc)
    except Exception as exc:
        logger.error("imap: unexpected error {}: {}", type(exc).__name__, exc)

    return results
