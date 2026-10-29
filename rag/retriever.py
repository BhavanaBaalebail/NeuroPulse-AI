import json
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_SESSION_DB = Path(__file__).resolve().parent.parent / "data" / "session_history.json"


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec) + 1e-12
    matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-12
    return np.dot(matrix, query_vec) / (matrix_norm * query_norm)


def _format_embedding_text(record: Dict[str, Any]) -> str:
    fp = record["fatigue_pattern"]
    return (
        f"fatigue level {record['fatigue_level']}, "
        f"EAR {fp['EAR']}, "
        f"MCR {fp['MCR']}, "
        f"ITSL {fp['ITSL']}, "
        f"time {record['time_of_day']}, "
        f"session {record['session_type']}"
    )


def _format_query_text(query: Dict[str, Any]) -> str:
    return (
        f"fatigue level {query['fatigue_level']}, "
        f"EAR {query['EAR']}, "
        f"MCR {query['MCR']}, "
        f"ITSL {query['ITSL']}, "
        f"time {query['time_of_day']}, "
        f"session {query['session_type']}"
    )


class SessionRetriever:
    def __init__(self, db_path: Path | None = None, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.db_path = db_path or DEFAULT_SESSION_DB
        self.model = SentenceTransformer(model_name)
        self.records = self._load_records()
        self._build_index()

    def _load_records(self) -> List[Dict[str, Any]]:
        if not self.db_path.exists():
            return []
        with self.db_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def _save_records(self) -> None:
        with self.db_path.open("w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def _build_index(self) -> None:
        if not self.records:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            return
        texts = [_format_embedding_text(r) for r in self.records]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def reload(self) -> None:
        self.records = self._load_records()
        self._build_index()

    def retrieve(self, query: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        if len(self.records) == 0:
            return []
        query_text = _format_query_text(query)
        query_vec = self.model.encode(query_text, convert_to_numpy=True, show_progress_bar=False)
        sims = _cosine_similarity(query_vec, self.embeddings)
        top_indices = np.argsort(-sims)[: max(1, top_k)]
        out: List[Dict[str, Any]] = []
        for idx in top_indices:
            rec = dict(self.records[int(idx)])
            rec["similarity"] = float(max(0.0, min(1.0, sims[int(idx)])))
            out.append(rec)
        return out

    def get_intervention_summary(self, sessions: List[Dict[str, Any]]) -> Dict[str, float]:
        grouped: Dict[str, List[float]] = {}
        for s in sessions:
            key = s.get("intervention_suggested", "unknown")
            grouped.setdefault(key, []).append(float(s.get("recovery_pct", 0.0)))
        return {k: round(float(sum(v) / max(1, len(v))), 1) for k, v in grouped.items()}

    def append_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        next_id = f"S{len(self.records) + 1:03d}"
        payload = dict(session)
        payload["session_id"] = next_id
        self.records.append(payload)
        self._save_records()
        self._build_index()
        return payload

    def update_session_intervention(
        self,
        session_id: str,
        intervention: str,
        intervention_taken: bool = True,
    ) -> bool:
        for rec in self.records:
            if rec.get("session_id") == session_id:
                rec["intervention_suggested"] = intervention
                rec["intervention_taken"] = intervention_taken
                self._save_records()
                return True
        return False


def format_retrieved_sessions_for_prompt(sessions: List[Dict[str, Any]]) -> str:
    if not sessions:
        return "No similar history found yet."
    lines = []
    for s in sessions:
        lines.append(
            f"• [{s['date']}] [{s['time_of_day']}]: Similar fatigue ({s['fatigue_level']}), "
            f"tried '{s['intervention_suggested']}' → {s['recovery_pct']}% recovery in "
            f"{s['time_to_recover_mins']} mins"
        )
    return "\n".join(lines)
