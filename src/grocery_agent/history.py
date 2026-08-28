"""Persistent conversation history owned by the local application."""

from __future__ import annotations

import sqlite3
import unicodedata
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

from .database import Database
from .models import Conversation, Message


class ConversationError(ValueError):
    """Base class for expected conversation-history errors."""


class ConversationNotFoundError(ConversationError):
    """Raised when a conversation does not belong to the requested user."""


class InvalidMessageError(ConversationError):
    """Raised when a role or message body is invalid."""


Clock = Callable[[], datetime]


def _system_clock() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_user_id(user_id: object) -> str:
    if not isinstance(user_id, str):
        raise ConversationError("User id must not be empty")
    normalized = " ".join(unicodedata.normalize("NFKC", user_id).split())
    if not normalized:
        raise ConversationError("User id must not be empty")
    return normalized


def _normalize_conversation_id(conversation_id: object) -> int:
    if (
        isinstance(conversation_id, bool)
        or not isinstance(conversation_id, int)
        or conversation_id <= 0
    ):
        raise ConversationNotFoundError(
            f"Conversation {conversation_id!r} was not found"
        )
    return conversation_id


def _normalize_limit(limit: object) -> Optional[int]:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ConversationError("Message limit must be a positive integer")
    return limit


class ConversationService:
    """Creates conversations and stores their user-visible messages."""

    def __init__(self, database: Database, clock: Clock = _system_clock) -> None:
        self.database = database
        self._clock = clock

    def start_conversation(self, user_id: str) -> Conversation:
        normalized_user_id = _normalize_user_id(user_id)
        now = self._now()
        with self.database.connect() as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO users (id, created_at) VALUES (?, ?)",
                (normalized_user_id, now),
            )
            cursor = connection.execute(
                "INSERT INTO conversations (user_id, created_at) VALUES (?, ?)",
                (normalized_user_id, now),
            )
            return Conversation(
                id=int(cursor.lastrowid),
                user_id=normalized_user_id,
                created_at=datetime.fromisoformat(now),
            )

    def get_conversation(
        self, user_id: str, conversation_id: int
    ) -> Conversation:
        normalized_user_id = _normalize_user_id(user_id)
        normalized_conversation_id = _normalize_conversation_id(conversation_id)
        with self.database.connect() as connection:
            row = self._owned_conversation_row(
                connection, normalized_user_id, normalized_conversation_id
            )
            return self._load_conversation(row)

    def append_message(
        self,
        user_id: str,
        conversation_id: int,
        role: str,
        content: str,
    ) -> Message:
        normalized_user_id = _normalize_user_id(user_id)
        normalized_conversation_id = _normalize_conversation_id(conversation_id)
        if role not in {"user", "assistant"}:
            raise InvalidMessageError("Role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise InvalidMessageError("Message content must not be empty")
        normalized_content = content.strip()
        now = self._now()

        with self.database.connect() as connection, connection:
            self._owned_conversation_row(
                connection, normalized_user_id, normalized_conversation_id
            )
            cursor = connection.execute(
                """
                INSERT INTO messages (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (normalized_conversation_id, role, normalized_content, now),
            )
            return Message(
                id=int(cursor.lastrowid),
                conversation_id=normalized_conversation_id,
                role=role,
                content=normalized_content,
                created_at=datetime.fromisoformat(now),
            )

    def list_messages(
        self,
        user_id: str,
        conversation_id: int,
        limit: Optional[int] = None,
    ) -> Tuple[Message, ...]:
        normalized_user_id = _normalize_user_id(user_id)
        normalized_conversation_id = _normalize_conversation_id(conversation_id)
        normalized_limit = _normalize_limit(limit)

        with self.database.connect() as connection:
            self._owned_conversation_row(
                connection, normalized_user_id, normalized_conversation_id
            )
            if normalized_limit is None:
                rows = connection.execute(
                    """
                    SELECT id, conversation_id, role, content, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id
                    """,
                    (normalized_conversation_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT id, conversation_id, role, content, created_at
                    FROM (
                        SELECT id, conversation_id, role, content, created_at
                        FROM messages
                        WHERE conversation_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id
                    """,
                    (normalized_conversation_id, normalized_limit),
                ).fetchall()
            return tuple(self._load_message(row) for row in rows)

    def _now(self) -> str:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ConversationError("Clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _owned_conversation_row(
        connection: sqlite3.Connection, user_id: str, conversation_id: int
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT id, user_id, created_at
            FROM conversations
            WHERE id = ? AND user_id = ?
            """,
            (conversation_id, user_id),
        ).fetchone()
        if row is None:
            raise ConversationNotFoundError(
                f"Conversation {conversation_id!r} was not found for this user"
            )
        return row

    @staticmethod
    def _load_conversation(row: sqlite3.Row) -> Conversation:
        return Conversation(
            id=row["id"],
            user_id=row["user_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _load_message(row: sqlite3.Row) -> Message:
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
