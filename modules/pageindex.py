import os
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class CacheEntry:
    url: str
    url_hash: str
    title: Optional[str]
    content: str
    extracted_at: datetime
    expires_at: datetime
    source: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "url_hash": self.url_hash,
            "title": self.title,
            "content": self.content,
            "extracted_at": self.extracted_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "source": self.source,
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "CacheEntry":
        return cls(
            url=row["url"],
            url_hash=row["url_hash"],
            title=row["title"],
            content=row["content"],
            extracted_at=datetime.fromisoformat(row["extracted_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            source=row["source"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


class PageIndex:
    """Persistent SQLite-based cache for extracted webpage content."""

    def __init__(
        self,
        db_path: str = "data/pageindex.db",
        default_ttl: int = 86400,
        max_size_mb: int = 100,
    ):
        self.db_path = Path(db_path)
        self.default_ttl = default_ttl
        self.max_size_mb = max_size_mb
        self._ensure_db()

    def _ensure_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pageindex (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    title TEXT,
                    content TEXT NOT NULL,
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    source TEXT DEFAULT 'jina',
                    metadata TEXT
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pageindex_time 
                ON pageindex(extracted_at)
            """
            )
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _hash_url(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    def get(self, url: str) -> Optional[CacheEntry]:
        url_hash = self._hash_url(url)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pageindex WHERE url_hash = ?",
                (url_hash,),
            ).fetchone()

            if row is None:
                return None

            entry = CacheEntry.from_row(row)
            if datetime.now() > entry.expires_at:
                conn.execute(
                    "DELETE FROM pageindex WHERE url_hash = ?",
                    (url_hash,),
                )
                conn.commit()
                return None

            return entry

    def set(
        self,
        url: str,
        content: str,
        title: Optional[str] = None,
        ttl: Optional[int] = None,
        source: str = "jina",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        url_hash = self._hash_url(url)
        ttl = ttl or self.default_ttl
        now = datetime.now()
        expires_at = now + timedelta(seconds=ttl)

        entry = CacheEntry(
            url=url,
            url_hash=url_hash,
            title=title,
            content=content,
            extracted_at=now,
            expires_at=expires_at,
            source=source,
            metadata=metadata or {},
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pageindex 
                (url_hash, url, title, content, extracted_at, expires_at, source, metadata)
                VALUES (:url_hash, :url, :title, :content, :extracted_at, :expires_at, :source, :metadata)
            """,
                entry.to_dict(),
            )
            conn.commit()

        self._enforce_size_limit()

    def _enforce_size_limit(self) -> None:
        db_size = self.db_path.stat().st_size / (1024 * 1024)
        if db_size > self.max_size_mb:
            with self._connect() as conn:
                conn.execute(
                    """
                    DELETE FROM pageindex 
                    WHERE url_hash IN (
                        SELECT url_hash FROM pageindex 
                        ORDER BY extracted_at ASC 
                        LIMIT (SELECT COUNT(*) / 10 FROM pageindex)
                    )
                """
                )
                conn.commit()

    def invalidate(self, url: str) -> bool:
        url_hash = self._hash_url(url)
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM pageindex WHERE url_hash = ?",
                (url_hash,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def invalidate_expired(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM pageindex WHERE expires_at < ?",
                (datetime.now().isoformat(),),
            )
            conn.commit()
            return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM pageindex"
            ).fetchone()[0]

            expired = conn.execute(
                "SELECT COUNT(*) FROM pageindex WHERE expires_at < ?",
                (datetime.now().isoformat(),),
            ).fetchone()[0]

            db_size = self.db_path.stat().st_size / (1024 * 1024)

            return {
                "total_entries": total,
                "expired_entries": expired,
                "db_size_mb": round(db_size, 2),
                "max_size_mb": self.max_size_mb,
                "default_ttl_seconds": self.default_ttl,
            }

    def clear(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM pageindex")
            conn.commit()
            return cursor.rowcount

    def close(self) -> None:
        pass
