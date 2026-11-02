"""Port of commands/vectorstore.rs.

Two interchangeable storage backends sit behind one small interface:

* ``LanceBackend``  - LanceDB (lazy ``import lancedb``), same on-disk location
  (``<project>/.llm-wiki/lancedb``), table names (``wiki_vectors`` v1,
  ``wiki_chunks_v2`` v2) and schemas as the Rust implementation, so existing
  indexes keep working.
* ``SqliteBackend`` - sqlite3 + (optional) numpy fallback, stored in
  ``<project>/.llm-wiki/vectors.sqlite3``. Used automatically when lancedb
  cannot be imported.

``LLM_WIKI_VECTOR_BACKEND=lancedb|sqlite`` forces one. Scoring is identical
for both: ``score = 1 / (1 + squared_L2_distance)`` (LanceDB's default L2
``_distance`` is the squared Euclidean distance).

Public Python helpers (same names as the Rust pub fns): every command below is
a plain synchronous function and can be called directly, plus
``validate_page_id_for_v2``, ``vector_upsert_chunks_with_revision`` and
``vector_page_revision_match``.
"""
from __future__ import annotations

import array
import heapq
import os
import sqlite3
import sys
import threading
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..errors import CommandError
from ..registry import command

try:  # optional accelerator
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None

TABLE_V1 = "wiki_vectors"
TABLE_V2 = "wiki_chunks_v2"
MAX_PAGE_ID_CHARS = 256

_U32_MAX = 0xFFFFFFFF

# ---------------------------------------------------------------------------
# page id validation
# ---------------------------------------------------------------------------

_DISALLOWED_RANGES = (
    (0x00AD, 0x00AD),
    (0x061C, 0x061C),
    (0x200B, 0x200F),
    (0x2028, 0x202E),
    (0x2060, 0x206F),
    (0xFEFF, 0xFEFF),
    (0xFFF9, 0xFFFB),
    (0xE0000, 0xE007F),
)


def _is_disallowed_page_id_char(c: str) -> bool:
    if unicodedata.category(c) == "Cc":  # Rust char::is_control
        return True
    if c in ("\\", "'", '"'):
        return True
    cp = ord(c)
    return any(lo <= cp <= hi for lo, hi in _DISALLOWED_RANGES)


def _rust_char_debug(c: str) -> str:
    """Approximation of Rust's `{:?}` for a char."""
    simple = {"\t": "\\t", "\r": "\\r", "\n": "\\n", "\0": "\\0", "\\": "\\\\", "'": "\\'", '"': '"'}
    if c in simple:
        return "'" + simple[c] + "'"
    cp = ord(c)
    if unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp", "Zs", "Co", "Cn") or cp == 0xAD:
        return "'\\u{%x}'" % cp
    return "'" + c + "'"


def validate_page_id_for_v2(page_id: str) -> None:
    """Raise CommandError when `page_id` is unsafe (mirrors validate_page_id_common)."""
    if not isinstance(page_id, str) or page_id == "":
        raise CommandError("Invalid page_id: empty or too long")
    if len(page_id) > MAX_PAGE_ID_CHARS:
        raise CommandError("Invalid page_id: empty or too long")
    for c in page_id:
        if _is_disallowed_page_id_char(c):
            raise CommandError(
                "Invalid page_id: contains disallowed character %s: %s" % (_rust_char_debug(c), page_id)
            )


validate_page_id = validate_page_id_for_v2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def db_path(project_path: str) -> str:
    return "%s/.llm-wiki/lancedb" % project_path.replace("\\", "/")


_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _v2_lock(project_path: str) -> threading.RLock:
    key = db_path(project_path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.RLock()
        return lock


def _to_f32_list(vec: Any, what: str = "embedding") -> List[float]:
    if not isinstance(vec, (list, tuple)):
        raise CommandError("invalid args: `%s` must be an array of numbers" % what)
    try:
        out = [float(x) for x in vec]
    except (TypeError, ValueError):
        raise CommandError("invalid args: `%s` must be an array of numbers" % what)
    return out


def _pack(vec: List[float]) -> bytes:
    try:
        return array.array("f", vec).tobytes()
    except OverflowError as e:
        raise CommandError("Batch error: %s" % e)


def _unpack(blob: bytes) -> "array.array":
    a = array.array("f")
    a.frombytes(blob)
    return a


def _check_top_k(top_k: Any) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise CommandError("invalid args: `topK` must be a non-negative integer")
    return top_k


# ---------------------------------------------------------------------------
# backend interface
# ---------------------------------------------------------------------------

class VectorBackend:
    name = "base"

    # v1 (legacy, one row per page)
    def v1_upsert(self, page_id: str, vec: List[float]) -> None: ...
    def v1_search(self, vec: List[float], top_k: int) -> List[Tuple[str, float]]: ...
    def v1_delete(self, page_id: str) -> None: ...
    def v1_count(self) -> int: ...
    def v1_drop(self) -> None: ...

    # v2 (chunks)
    def v2_replace_page(self, page_id: str, rows: List[Dict[str, Any]], dim: int) -> None: ...
    def v2_search(self, vec: List[float], top_k: int) -> List[Dict[str, Any]]: ...
    def v2_delete_page(self, page_id: str) -> None: ...
    def v2_count(self, page_id: Optional[str] = None) -> int: ...
    def v2_drop(self) -> None: ...
    def v2_optimize(self) -> None: ...


# ---------------------------------------------------------------------------
# sqlite backend
# ---------------------------------------------------------------------------

class SqliteBackend(VectorBackend):
    name = "sqlite"

    def __init__(self, project_path: str):
        self.path = Path(project_path) / ".llm-wiki" / "vectors.sqlite3"

    def _connect(self, create: bool) -> Optional[sqlite3.Connection]:
        if not self.path.exists():
            if not create:
                return None
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(str(self.path), timeout=30)
        except sqlite3.Error as e:
            raise CommandError("DB connect error: %s" % e)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS legacy_vectors (
                page_id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, page_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL, heading_path TEXT NOT NULL,
                dim INTEGER NOT NULL, vector BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS chunks_page ON chunks(page_id);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        return conn

    # -- generic helpers --
    @staticmethod
    def _table_dim(conn, table: str) -> Optional[int]:
        row = conn.execute("SELECT dim FROM %s LIMIT 1" % table).fetchone()
        return row[0] if row else None

    @staticmethod
    def _nearest(conn, table: str, vec: List[float], top_k: int) -> List[Tuple[int, float]]:
        dim = len(vec)
        if top_k == 0:
            return []
        cur = conn.execute("SELECT rowid, dim, vector FROM %s" % table)
        if _np is not None:
            ids: List[int] = []
            blobs: List[bytes] = []
            for rowid, d, blob in cur:
                if d != dim:
                    raise CommandError(
                        "Execute search error: query vector dimension %d does not match table dimension %d"
                        % (dim, d)
                    )
                ids.append(rowid)
                blobs.append(blob)
            if not ids:
                return []
            m = _np.frombuffer(b"".join(blobs), dtype=_np.float32).reshape(len(ids), dim)
            q = _np.asarray(vec, dtype=_np.float32)
            diff = m - q
            dist = (diff * diff).sum(axis=1)
            k = min(top_k, len(ids))
            order = _np.argsort(dist, kind="stable")[:k]
            return [(ids[int(i)], float(dist[int(i)])) for i in order]

        q32 = _unpack(_pack(vec))
        heap: List[Tuple[float, int]] = []
        for rowid, d, blob in cur:
            if d != dim:
                raise CommandError(
                    "Execute search error: query vector dimension %d does not match table dimension %d"
                    % (dim, d)
                )
            v = _unpack(blob)
            dist = 0.0
            for a, b in zip(v, q32):
                t = a - b
                dist += t * t
            heap.append((dist, rowid))
        return [(rid, dist) for dist, rid in heapq.nsmallest(top_k, heap)]

    # -- v1 --
    def v1_upsert(self, page_id, vec):
        conn = self._connect(True)
        try:
            with conn:
                existing = self._table_dim(conn, "legacy_vectors")
                if existing is not None and existing != len(vec):
                    raise CommandError(
                        "Add error: vector dimension %d does not match table dimension %d" % (len(vec), existing)
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO legacy_vectors(page_id, dim, vector) VALUES (?,?,?)",
                    (page_id, len(vec), _pack(vec)),
                )
        finally:
            conn.close()

    def v1_search(self, vec, top_k):
        conn = self._connect(False)
        if conn is None:
            return []
        try:
            hits = self._nearest(conn, "legacy_vectors", vec, top_k)
            out = []
            for rowid, dist in hits:
                (pid,) = conn.execute("SELECT page_id FROM legacy_vectors WHERE rowid=?", (rowid,)).fetchone()
                out.append((pid, dist))
            return out
        finally:
            conn.close()

    def v1_delete(self, page_id):
        conn = self._connect(False)
        if conn is None:
            return
        try:
            with conn:
                conn.execute("DELETE FROM legacy_vectors WHERE page_id=?", (page_id,))
        finally:
            conn.close()

    def v1_count(self):
        conn = self._connect(False)
        if conn is None:
            return 0
        try:
            return conn.execute("SELECT COUNT(*) FROM legacy_vectors").fetchone()[0]
        finally:
            conn.close()

    def v1_drop(self):
        conn = self._connect(False)
        if conn is None:
            return
        try:
            with conn:
                conn.execute("DELETE FROM legacy_vectors")
        finally:
            conn.close()

    # -- v2 --
    def v2_replace_page(self, page_id, rows, dim):
        conn = self._connect(True)
        try:
            with conn:  # single transaction: atomic page replace
                existing = self._table_dim(conn, "chunks")
                if existing is not None and existing != dim:
                    other = conn.execute("SELECT 1 FROM chunks WHERE page_id<>? LIMIT 1", (page_id,)).fetchone()
                    if other:
                        raise CommandError(
                            "Atomic page upsert failed: vector dimension %d does not match table dimension %d"
                            % (dim, existing)
                        )
                conn.execute("DELETE FROM chunks WHERE page_id=?", (page_id,))
                conn.executemany(
                    "INSERT OR REPLACE INTO chunks(chunk_id,page_id,chunk_index,chunk_text,heading_path,dim,vector)"
                    " VALUES (?,?,?,?,?,?,?)",
                    [
                        (
                            r["chunk_id"], page_id, r["chunk_index"], r["chunk_text"],
                            r["heading_path"], dim, _pack(r["embedding"]),
                        )
                        for r in rows
                    ],
                )
        finally:
            conn.close()

    def v2_search(self, vec, top_k):
        conn = self._connect(False)
        if conn is None:
            return []
        try:
            out = []
            for rowid, dist in self._nearest(conn, "chunks", vec, top_k):
                cid, pid, idx, text, hp = conn.execute(
                    "SELECT chunk_id,page_id,chunk_index,chunk_text,heading_path FROM chunks WHERE rowid=?",
                    (rowid,),
                ).fetchone()
                out.append({"chunk_id": cid, "page_id": pid, "chunk_index": idx, "chunk_text": text,
                            "heading_path": hp, "distance": dist})
            return out
        finally:
            conn.close()

    def v2_delete_page(self, page_id):
        conn = self._connect(False)
        if conn is None:
            return
        try:
            with conn:
                conn.execute("DELETE FROM chunks WHERE page_id=?", (page_id,))
        finally:
            conn.close()

    def v2_count(self, page_id=None):
        conn = self._connect(False)
        if conn is None:
            return 0
        try:
            if page_id is None:
                return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM chunks WHERE page_id=?", (page_id,)).fetchone()[0]
        finally:
            conn.close()

    def v2_drop(self):
        conn = self._connect(False)
        if conn is None:
            return
        try:
            with conn:
                conn.execute("DELETE FROM chunks")
        finally:
            conn.close()

    def v2_optimize(self):
        conn = self._connect(False)
        if conn is None:
            return
        try:
            conn.execute("VACUUM")
        except sqlite3.Error as e:
            raise CommandError("Compact chunks error: %s" % e)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# LanceDB backend
# ---------------------------------------------------------------------------

class LanceBackend(VectorBackend):
    name = "lancedb"

    def __init__(self, project_path: str):
        import lancedb  # noqa: F401  (lazy; ImportError -> caller falls back)
        import pyarrow  # noqa: F401

        self._lancedb = lancedb
        self._pa = pyarrow
        self._uri = db_path(project_path)

    def _db(self):
        try:
            Path(self._uri).mkdir(parents=True, exist_ok=True)
            return self._lancedb.connect(self._uri)
        except Exception as e:
            raise CommandError("DB connect error: %s" % e)

    @staticmethod
    def _names(db) -> List[str]:
        try:
            res = db.table_names()
        except Exception as e:
            raise CommandError("List tables error: %s" % e)
        if hasattr(res, "tables"):  # newer ListTablesResponse
            res = res.tables
        return list(res)

    def _vec_type(self, dim: int):
        pa = self._pa
        return pa.list_(pa.field("item", pa.float32(), nullable=True), dim)

    def _schema_v1(self, dim):
        pa = self._pa
        return pa.schema([
            pa.field("page_id", pa.string(), nullable=False),
            pa.field("vector", self._vec_type(dim), nullable=False),
        ])

    def _schema_v2(self, dim):
        pa = self._pa
        return pa.schema([
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("page_id", pa.string(), nullable=False),
            pa.field("chunk_index", pa.uint32(), nullable=False),
            pa.field("chunk_text", pa.string(), nullable=False),
            pa.field("heading_path", pa.string(), nullable=False),
            pa.field("vector", self._vec_type(dim), nullable=False),
        ])

    def _open(self, db, name: str):
        try:
            return db.open_table(name)
        except Exception as e:
            raise CommandError("Open table error: %s" % e)

    def _search(self, table, vec, top_k, err_prefix="Execute search error"):
        try:
            return table.search(vec).limit(top_k).to_list()
        except Exception as e:
            raise CommandError("%s: %s" % (err_prefix, e))

    @staticmethod
    def _count(table, filt=None) -> int:
        try:
            return int(table.count_rows(filt) if filt else table.count_rows())
        except Exception as e:
            raise CommandError("Count error: %s" % e)

    def _drop(self, db, name, msg):
        try:
            db.drop_table(name)
        except Exception as e:
            raise CommandError("%s: %s" % (msg, e))

    # v1
    def v1_upsert(self, page_id, vec):
        pa = self._pa
        dim = len(vec)
        schema = self._schema_v1(dim)
        try:
            data = pa.table(
                {"page_id": pa.array([page_id], pa.string()),
                 "vector": pa.array([vec], type=self._vec_type(dim))},
                schema=schema,
            )
        except Exception as e:
            raise CommandError("Batch error: %s" % e)
        db = self._db()
        if TABLE_V1 in self._names(db):
            table = self._open(db, TABLE_V1)
            try:
                table.delete("page_id = '%s'" % page_id)
            except Exception as e:
                print("[vectorstore] Warning: delete before upsert failed for '%s': %s" % (page_id, e),
                      file=sys.stderr)
            try:
                table.add(data)
            except Exception as e:
                raise CommandError("Add error: %s" % e)
        else:
            try:
                db.create_table(TABLE_V1, data, schema=schema)
            except Exception as e:
                raise CommandError("Create table error: %s" % e)

    def v1_search(self, vec, top_k):
        db = self._db()
        if TABLE_V1 not in self._names(db):
            return []
        table = self._open(db, TABLE_V1)
        rows = self._search(table, vec, top_k)
        return [(r["page_id"], float(r["_distance"])) for r in rows]

    def v1_delete(self, page_id):
        db = self._db()
        if TABLE_V1 not in self._names(db):
            return
        table = self._open(db, TABLE_V1)
        try:
            table.delete("page_id = '%s'" % page_id)
        except Exception as e:
            raise CommandError("Delete error: %s" % e)

    def v1_count(self):
        db = self._db()
        if TABLE_V1 not in self._names(db):
            return 0
        return self._count(self._open(db, TABLE_V1))

    def v1_drop(self):
        db = self._db()
        if TABLE_V1 not in self._names(db):
            return
        self._drop(db, TABLE_V1, "Drop table error")

    # v2
    def _batch_v2(self, page_id, rows, dim):
        pa = self._pa
        schema = self._schema_v2(dim)
        try:
            return pa.table(
                {
                    "chunk_id": pa.array([r["chunk_id"] for r in rows], pa.string()),
                    "page_id": pa.array([page_id] * len(rows), pa.string()),
                    "chunk_index": pa.array([r["chunk_index"] for r in rows], pa.uint32()),
                    "chunk_text": pa.array([r["chunk_text"] for r in rows], pa.string()),
                    "heading_path": pa.array([r["heading_path"] for r in rows], pa.string()),
                    "vector": pa.array([r["embedding"] for r in rows], type=self._vec_type(dim)),
                },
                schema=schema,
            ), schema
        except Exception as e:
            raise CommandError("Batch error: %s" % e)

    def v2_replace_page(self, page_id, rows, dim):
        batch, schema = self._batch_v2(page_id, rows, dim)
        db = self._db()
        if TABLE_V2 in self._names(db):
            table = self._open(db, TABLE_V2)
            try:
                (table.merge_insert("chunk_id")
                 .when_matched_update_all()
                 .when_not_matched_insert_all()
                 .when_not_matched_by_source_delete("page_id = '%s'" % page_id)
                 .execute(batch))
            except Exception as e:
                raise CommandError("Atomic page upsert failed: %s" % e)
        else:
            try:
                db.create_table(TABLE_V2, batch, schema=schema)
            except Exception as e:
                raise CommandError("Create table error: %s" % e)

    def v2_search(self, vec, top_k):
        db = self._db()
        if TABLE_V2 not in self._names(db):
            return []
        table = self._open(db, TABLE_V2)
        rows = self._search(table, vec, top_k)
        return [
            {"chunk_id": r["chunk_id"], "page_id": r["page_id"], "chunk_index": int(r["chunk_index"]),
             "chunk_text": r["chunk_text"], "heading_path": r["heading_path"],
             "distance": float(r["_distance"])}
            for r in rows
        ]

    def v2_delete_page(self, page_id):
        db = self._db()
        if TABLE_V2 not in self._names(db):
            return
        try:
            self._open(db, TABLE_V2).delete("page_id = '%s'" % page_id)
        except CommandError:
            raise
        except Exception as e:
            raise CommandError("Delete error: %s" % e)

    def v2_count(self, page_id=None):
        db = self._db()
        if TABLE_V2 not in self._names(db):
            return 0
        table = self._open(db, TABLE_V2)
        if page_id is None:
            return self._count(table)
        try:
            return int(table.count_rows("page_id = '%s'" % page_id))
        except Exception as e:
            raise CommandError("Count page chunks error: %s" % e)

    def v2_drop(self):
        db = self._db()
        if TABLE_V2 not in self._names(db):
            return
        self._drop(db, TABLE_V2, "Drop chunk table error")

    def v2_optimize(self):
        from datetime import timedelta

        db = self._db()
        if TABLE_V2 not in self._names(db):
            return
        table = self._open(db, TABLE_V2)
        # Compact, then prune versions older than the current snapshot while
        # leaving delete_unverified disabled (unsafe under cross-process use).
        try:
            table.optimize(cleanup_older_than=timedelta(seconds=0), delete_unverified=False)
        except TypeError:
            try:
                table.compact_files()
            except Exception as e:
                raise CommandError("Compact chunks error: %s" % e)
            try:
                table.cleanup_old_versions(older_than=timedelta(seconds=0))
            except Exception as e:
                raise CommandError("Prune old chunk versions error: %s" % e)
        except Exception as e:
            raise CommandError("Compact chunks error: %s" % e)


# ---------------------------------------------------------------------------
# backend selection
# ---------------------------------------------------------------------------

def _lancedb_available() -> bool:
    try:
        import lancedb  # noqa: F401
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def backend_name() -> str:
    """Name of the backend that will be used right now: 'lancedb' or 'sqlite'."""
    forced = (os.environ.get("LLM_WIKI_VECTOR_BACKEND") or "").strip().lower()
    if forced in ("sqlite", "sqlite3"):
        return "sqlite"
    if forced == "lancedb":
        return "lancedb"
    return "lancedb" if _lancedb_available() else "sqlite"


def get_backend(project_path: str) -> VectorBackend:
    if backend_name() == "lancedb":
        try:
            return LanceBackend(project_path)
        except ImportError as e:
            raise CommandError("DB connect error: lancedb backend requested but not importable: %s" % e)
    return SqliteBackend(project_path)


# ---------------------------------------------------------------------------
# v1 commands
# ---------------------------------------------------------------------------

@command()
def vector_upsert(project_path: str, page_id: str, embedding: list) -> None:
    validate_page_id(page_id)
    vec = _to_f32_list(embedding)
    get_backend(project_path).v1_upsert(page_id, vec)
    return None


@command()
def vector_search(project_path: str, query_embedding: list, top_k: int) -> list:
    top_k = _check_top_k(top_k)
    vec = _to_f32_list(query_embedding, "queryEmbedding")
    hits = get_backend(project_path).v1_search(vec, top_k)
    return [{"page_id": pid, "score": 1.0 / (1.0 + d)} for pid, d in hits]


@command()
def vector_delete(project_path: str, page_id: str) -> None:
    validate_page_id(page_id)
    get_backend(project_path).v1_delete(page_id)
    return None


@command()
def vector_count(project_path: str) -> int:
    return get_backend(project_path).v1_count()


# ---------------------------------------------------------------------------
# v2 commands
# ---------------------------------------------------------------------------

def _parse_chunks(chunks: Any) -> List[Dict[str, Any]]:
    if not isinstance(chunks, (list, tuple)):
        raise CommandError("invalid args: `chunks` must be an array")
    out = []
    for c in chunks:
        if not isinstance(c, dict):
            raise CommandError("invalid args: each chunk must be an object")
        norm: Dict[str, Any] = {}
        for snake, camel in (("chunk_index", "chunkIndex"), ("chunk_text", "chunkText"),
                             ("heading_path", "headingPath"), ("embedding", "embedding")):
            if snake in c:
                norm[snake] = c[snake]
            elif camel in c:
                norm[snake] = c[camel]
            else:
                raise CommandError("invalid args: missing field `%s`" % snake)
        idx = norm["chunk_index"]
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx <= _U32_MAX:
            raise CommandError("invalid args: `chunk_index` must be an unsigned 32-bit integer")
        if not isinstance(norm["chunk_text"], str) or not isinstance(norm["heading_path"], str):
            raise CommandError("invalid args: `chunk_text` and `heading_path` must be strings")
        norm["embedding"] = _to_f32_list(norm["embedding"])
        out.append(norm)
    return out


def _upsert_chunks_inner(project_path: str, page_id: str, chunks: Any, fingerprint: Optional[str]) -> None:
    validate_page_id_for_v2(page_id)
    rows = _parse_chunks(chunks)
    with _v2_lock(project_path):
        if not rows:
            return
        dim = len(rows[0]["embedding"])
        if dim == 0:
            raise CommandError("Chunk #0 has empty embedding")
        for r in rows:
            if len(r["embedding"]) != dim:
                raise CommandError(
                    "Chunk #%d has embedding dim %d but batch dim is %d"
                    % (r["chunk_index"], len(r["embedding"]), dim)
                )
            r["chunk_id"] = "%s#%d" % (page_id, r["chunk_index"])
        get_backend(project_path).v2_replace_page(page_id, rows, dim)

        from . import page_embedding as pe

        if fingerprint is not None:
            try:
                pe.save_revision(project_path, page_id, fingerprint)
            except Exception as err:
                try:
                    pe.invalidate_page_revision(project_path, page_id)
                except Exception:
                    pass
                print("[page-embedding] failed to save revision metadata: %s" % err, file=sys.stderr)
        else:
            try:
                pe.invalidate_page_revision(project_path, page_id)
            except Exception:
                pass


@command()
def vector_upsert_chunks(project_path: str, page_id: str, chunks: list) -> None:
    """Replace all chunks of `page_id`. Empty `chunks` is a no-op (does not clear)."""
    _upsert_chunks_inner(project_path, page_id, chunks, None)
    return None


def vector_upsert_chunks_with_revision(project_path: str, page_id: str, chunks: list, fingerprint: str) -> None:
    """Like vector_upsert_chunks but records `fingerprint` as the page's embedding revision."""
    _upsert_chunks_inner(project_path, page_id, chunks, fingerprint)


def vector_page_revision_match(project_path: str, page_id: str, fingerprint: str) -> Optional[int]:
    """Chunk count if the stored revision equals `fingerprint` and chunks exist, else None."""
    validate_page_id_for_v2(page_id)
    with _v2_lock(project_path):
        from . import page_embedding as pe

        if pe.load_revision(project_path, page_id) != fingerprint:
            return None
        count = get_backend(project_path).v2_count(page_id)
        return count if count > 0 else None


@command()
def vector_search_chunks(project_path: str, query_embedding: list, top_k: int) -> list:
    """Top-K chunks: dicts {chunk_id,page_id,chunk_index,chunk_text,heading_path,score}."""
    top_k = _check_top_k(top_k)
    vec = _to_f32_list(query_embedding, "queryEmbedding")
    with _v2_lock(project_path):
        rows = get_backend(project_path).v2_search(vec, top_k)
    out = []
    for r in rows:
        d = r.pop("distance")
        r["score"] = 1.0 / (1.0 + d)
        out.append(r)
    return out


@command()
def vector_delete_page(project_path: str, page_id: str) -> None:
    validate_page_id_for_v2(page_id)
    with _v2_lock(project_path):
        get_backend(project_path).v2_delete_page(page_id)
    return None


@command()
def vector_count_chunks(project_path: str) -> int:
    with _v2_lock(project_path):
        return get_backend(project_path).v2_count()


@command()
def vector_clear_chunks(project_path: str) -> None:
    with _v2_lock(project_path):
        get_backend(project_path).v2_drop()
    return None


@command()
def vector_optimize_chunks(project_path: str) -> None:
    with _v2_lock(project_path):
        get_backend(project_path).v2_optimize()
    return None


@command()
def vector_legacy_row_count(project_path: str) -> int:
    return get_backend(project_path).v1_count()


@command()
def vector_drop_legacy(project_path: str) -> None:
    get_backend(project_path).v1_drop()
    return None
