"""Server-side replacement for src/stores/review-store.ts + src/lib/review-utils.ts + sweep-reviews.ts.

The Zustand review store is gone: review items live in ``<project>/.llm-wiki/review.json``
(a JSON array of ReviewItem, exactly what ``commands/rest_api.py`` and the desktop app read).
All mutations are read-modify-write under a per-file lock and written atomically.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence

from .. import events

REVIEW_UPDATED_TOPIC = "ingest://reviews-updated"

_locks: Dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()

_PREFIX_RE = re.compile(
    r"^(missing[\s-]?page[:：]\s*|duplicate[\s-]?page[:：]\s*|possible[\s-]?duplicate[:：]\s*"
    r"|缺失页面[:：]\s*|缺少页面[:：]\s*|重复页面[:：]\s*|疑似重复[:：]\s*)", re.I)


def normalize_review_title(title: str) -> str:
    """review-utils.ts normalizeReviewTitle."""
    t = _PREFIX_RE.sub("", title.lstrip(), count=1)
    return re.sub(r"\s+", " ", t).strip().lower()


def review_id_for(item: Dict[str, Any]) -> str:
    """FNV-1a over UTF-16 code units of ``type::normalizedTitle`` (review-store.ts reviewIdFor)."""
    key = "%s::%s" % (item.get("type", ""), normalize_review_title(str(item.get("title", ""))))
    h = 0x811C9DC5
    data = key.encode("utf-16-le")
    for i in range(0, len(data), 2):
        h ^= data[i] | (data[i + 1] << 8)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return "review-%08x" % h


def review_path(project_path: str) -> str:
    return os.path.join(str(project_path).replace("\\", "/"), ".llm-wiki", "review.json")


def _lock_for(path: str) -> threading.RLock:
    with _locks_guard:
        return _locks.setdefault(path, threading.RLock())


def _union(a: Optional[List[str]], b: Optional[List[str]]) -> Optional[List[str]]:
    merged: List[str] = []
    for v in list(a or []) + list(b or []):
        if v not in merged:
            merged.append(v)
    return merged or None


def _merge_options(a: List[dict], b: List[dict]) -> List[dict]:
    by_action: Dict[Any, dict] = {}
    for o in list(a) + list(b):
        by_action[o.get("action")] = o
    return list(by_action.values())


def _merge_items(a: dict, b: dict) -> dict:
    resolved = bool(a.get("resolved") or b.get("resolved"))
    out = dict(a)
    out["resolved"] = resolved
    if resolved:
        ra = a.get("resolvedAction") or b.get("resolvedAction")
        if ra is not None:
            out["resolvedAction"] = ra
    else:
        out.pop("resolvedAction", None)
    out["description"] = a.get("description") or b.get("description") or ""
    sp = a.get("sourcePath") if a.get("sourcePath") is not None else b.get("sourcePath")
    if sp is not None:
        out["sourcePath"] = sp
    for key in ("affectedPages", "searchQueries"):
        u = _union(a.get(key), b.get(key))
        if u is None:
            out.pop(key, None)
        else:
            out[key] = u
    out["options"] = _merge_options(a.get("options") or [], b.get("options") or [])
    out["createdAt"] = min(a.get("createdAt", 0), b.get("createdAt", 0))
    return out


def normalize_review_items(items: Sequence[dict]) -> List[dict]:
    by_id: Dict[str, dict] = {}
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["id"] = review_id_for(item)
        by_id[item["id"]] = _merge_items(by_id[item["id"]], item) if item["id"] in by_id else item
    return list(by_id.values())


def load_items(project_path: str) -> List[dict]:
    try:
        with open(review_path(project_path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    return [i for i in data if isinstance(i, dict)] if isinstance(data, list) else []


def _write_items(project_path: str, items: List[dict]) -> None:
    path = review_path(project_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.%s.tmp" % (path, uuid.uuid4().hex[:8])
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(items, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _notify(project_path: str, count: int) -> None:
    try:
        events.publish(REVIEW_UPDATED_TOPIC, {"projectPath": str(project_path), "count": count})
    except Exception:
        pass


def add_items(project_path: str, new_items: Sequence[dict]) -> List[dict]:
    """review-store.ts ``addItems``: dedupe on the content-stable id against ALL existing
    items (resolved ones included, so resolved state survives regeneration)."""
    if not new_items:
        return []
    path = review_path(project_path)
    with _lock_for(path):
        result = load_items(project_path)
        index_by_id: Dict[str, int] = {}
        for idx, it in enumerate(result):
            # Items written by older builds may carry non-stable ids; match on both.
            index_by_id[review_id_for(it)] = idx
            if isinstance(it.get("id"), str):
                index_by_id.setdefault(it["id"], idx)
        added: List[dict] = []
        for incoming in new_items:
            rid = review_id_for(incoming)
            idx = index_by_id.get(rid)
            if idx is not None:
                old = result[idx]
                merged = dict(old)
                merged["description"] = incoming.get("description") or old.get("description", "")
                sp = incoming.get("sourcePath") if incoming.get("sourcePath") is not None else old.get("sourcePath")
                if sp is not None:
                    merged["sourcePath"] = sp
                for key in ("affectedPages", "searchQueries"):
                    u = _union(old.get(key), incoming.get(key))
                    if u is None:
                        merged.pop(key, None)
                    else:
                        merged[key] = u
                result[idx] = merged
            else:
                item = {k: v for k, v in incoming.items() if v is not None}
                item.update({"id": rid, "resolved": False, "createdAt": int(time.time() * 1000)})
                item.setdefault("options", [])
                result.append(item)
                index_by_id[rid] = len(result) - 1
                added.append(item)
        _write_items(project_path, result)
    _notify(project_path, len(result))
    return added


def resolve_item(project_path: str, item_id: str, action: str) -> bool:
    path = review_path(project_path)
    with _lock_for(path):
        items = load_items(project_path)
        hit = False
        for it in items:
            if it.get("id") == item_id or review_id_for(it) == item_id:
                it["resolved"] = True
                it["resolvedAction"] = action
                hit = True
        if hit:
            _write_items(project_path, items)
    if hit:
        _notify(project_path, len(items))
    return hit


def resolve_many(project_path: str, ids: Sequence[str], action: str) -> int:
    wanted = set(ids)
    if not wanted:
        return 0
    path = review_path(project_path)
    with _lock_for(path):
        items = load_items(project_path)
        n = 0
        for it in items:
            if (it.get("id") in wanted or review_id_for(it) in wanted) and not it.get("resolved"):
                it["resolved"] = True
                it["resolvedAction"] = action
                n += 1
        if n:
            _write_items(project_path, items)
    if n:
        _notify(project_path, len(items))
    return n


# --------------------------------------------------------------------------
# sweep-reviews.ts
# --------------------------------------------------------------------------

JUDGE_BATCH_SIZE = 40
MAX_JUDGE_BATCHES = 5
MAX_PAGES_IN_PROMPT = 300


def _flatten_md(directory: str) -> List[str]:
    out: List[str] = []
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(".md"):
                out.append(os.path.join(root, name))
    return sorted(out)


def build_wiki_index(project_path: str) -> Dict[str, Any]:
    from .frontmatter import parse_frontmatter

    pp = str(project_path).replace("\\", "/").rstrip("/")
    by_id: set = set()
    by_title: set = set()
    pages: List[dict] = []
    for fp in _flatten_md(pp + "/wiki"):
        pid = os.path.basename(fp)[:-3].lower()
        by_id.add(pid)
        title = None
        try:
            with open(fp, "r", encoding="utf-8") as f:
                fm = parse_frontmatter(f.read()).frontmatter
            t = fm.get("title") if fm else None
            if isinstance(t, str) and t.strip():
                title = t.strip()
                by_title.add(title.lower())
        except (OSError, ValueError):
            pass
        pages.append({"id": pid, "title": title})
    return {"byId": by_id, "byTitle": by_title, "pages": pages}


def _candidate_names(item: dict) -> List[str]:
    names: List[str] = []
    cleaned = normalize_review_title(item.get("title", ""))
    if cleaned and len(cleaned) <= 100:
        names.append(cleaned)
    for page in item.get("affectedPages") or []:
        base = re.sub(r"\.md$", "", page.split("/")[-1])
        if base and base.lower() not in names:
            names.append(base.lower())
    return names


def _page_exists(name: str, index: Dict[str, Any]) -> bool:
    n = name.strip().lower()
    if not n:
        return False
    return n in index["byId"] or re.sub(r"\s+", "-", n) in index["byId"] or n in index["byTitle"]


def extract_json_object(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.I).strip()
    start = text.find("{")
    if start == -1:
        return ""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


async def _judge_batch(batch: List[dict], index: Dict[str, Any], llm_config: dict, signal) -> set:
    from . import pipeline

    if not batch or (signal is not None and signal.aborted):
        return set()
    pages = index["pages"][:MAX_PAGES_IN_PROMPT]
    page_list = "\n".join(("- %s  (title: %s)" % (p["id"], p["title"])) if p["title"] else "- %s" % p["id"] for p in pages)
    lines = []
    for r in batch:
        affected = (" | affected: " + ", ".join(r["affectedPages"])) if r.get("affectedPages") else ""
        desc = (" — " + r["description"][:200]) if r.get("description") else ""
        lines.append('- id=%s [%s] "%s"%s%s' % (r["id"], r["type"], r["title"], desc, affected))
    prompt = "\n".join([
        "You are cleaning up a stale review queue for a personal wiki.",
        "After recent ingests, some review items may no longer be valid because the missing page now exists, the duplicate was resolved, or the referenced concept has been added.",
        "",
        "Current wiki pages (filename, optional title):",
        page_list or "(no pages yet)",
        "",
        "Pending review items to judge:",
        "\n".join(lines),
        "",
        "For each review item, decide whether the underlying condition has been RESOLVED by the current wiki state.",
        "Be conservative: only mark as resolved if you are confident the concern no longer applies.",
        "For contradictions, confirmations, or human-judgment items, default to keeping them pending.",
        "",
        'Respond with ONLY a JSON object in this exact shape: {"resolved": ["id1", "id2"]}',
        'If none of the items are resolved, return exactly: {"resolved": []}',
        "Do not wrap in markdown fences. Do not add commentary.",
    ])
    try:
        res = await pipeline.stream_text(llm_config, [{"role": "user", "content": prompt}], signal)
    except Exception:
        return set()
    if res.error or (signal is not None and signal.aborted) or not res.text.strip():
        return set()
    try:
        cleaned = extract_json_object(res.text)
        if not cleaned:
            return set()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("resolved"), list):
            return set()
        valid = {i["id"] for i in batch}
        return {i for i in parsed["resolved"] if isinstance(i, str) and i in valid}
    except ValueError:
        return set()


async def sweep_resolved_reviews(project_path: str, project_id: Optional[str] = None, signal=None) -> int:
    """Auto-resolve stale review items after the ingest queue drained (rules, then LLM judgment)."""
    from . import pipeline

    if signal is not None and signal.aborted:
        return 0
    pending = [i for i in load_items(project_path) if not i.get("resolved")]
    if not pending:
        return 0
    index = await asyncio.to_thread(build_wiki_index, project_path)
    rule_ids: List[str] = []
    still: List[dict] = []
    for item in pending:
        if signal is not None and signal.aborted:
            return 0
        hit = False
        if item.get("type") == "missing-page":
            names = _candidate_names(item)
            hit = bool(names) and any(_page_exists(n, index) for n in names)
        elif item.get("type") == "duplicate":
            affected = item.get("affectedPages") or []
            if affected:
                def exists(p: str) -> bool:
                    base = re.sub(r"\.md$", "", p.split("/")[-1]).lower()
                    return bool(base) and base in index["byId"]

                hit = not all(exists(p) for p in affected)
        if hit:
            rule_ids.append(item["id"])
        else:
            still.append(item)
    llm_ids: set = set()
    if still and not (signal is not None and signal.aborted):
        cfg = pipeline.get_task_llm_config(project_id) if project_id else None
        if cfg is not None and pipeline.has_usable_llm(cfg):
            remaining = list(still)
            batches = 0
            while remaining and batches < MAX_JUDGE_BATCHES:
                if signal is not None and signal.aborted:
                    break
                batch, remaining = remaining[:JUDGE_BATCH_SIZE], remaining[JUDGE_BATCH_SIZE:]
                got = await _judge_batch(batch, index, cfg, signal)
                batches += 1
                if not got:
                    break
                llm_ids |= got
    if not (signal is not None and signal.aborted):
        resolve_many(project_path, rule_ids, "auto-resolved")
        resolve_many(project_path, sorted(llm_ids), "llm-judged")
    return len(rule_ids) + len(llm_ids)
