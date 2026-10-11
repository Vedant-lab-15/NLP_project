"""
pipeline.py
-----------
Lightweight dataset loader and pipeline runner for MedExtract-Lite.

Notes file format (clinical_notes.txt):
  Notes are separated by a line containing "---".
  Each note optionally starts with "NOTE_ID: <id>".
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import pandas as pd

from src.nlp_engine import MedExtractEngine


def load_notes(filepath: str) -> List[Dict[str, str]]:
    """
    Parse clinical_notes.txt into a list of dicts.

    Each dict: {"note_id": str, "text": str}

    Raises FileNotFoundError if the file is missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Notes file not found: {filepath}\n"
            "Run: python data/generate_data.py"
        )

    with open(filepath, "r", encoding="utf-8") as fh:
        raw = fh.read()

    # Split on "---" separator lines
    chunks = re.split(r"\n---\n", raw)

    notes: List[Dict[str, str]] = []
    for i, chunk in enumerate(chunks, start=1):
        chunk = chunk.strip()
        if not chunk:
            continue
        id_match = re.search(r"NOTE_ID:\s*(\w+)", chunk)
        note_id = id_match.group(1) if id_match else str(i)
        notes.append({"note_id": note_id, "text": chunk})

    return notes


class ClinicalPipeline:
    """
    Ties note loading + NER extraction together.

    Parameters
    ----------
    engine     : MedExtractEngine
    notes_path : str  — path to clinical_notes.txt
    """

    def __init__(self, engine: MedExtractEngine, notes_path: str):
        self.engine = engine
        self.notes_path = notes_path
        self._notes: Optional[List[Dict[str, str]]] = None

    def load(self) -> "ClinicalPipeline":
        """Load notes from disk. Returns self for chaining."""
        self._notes = load_notes(self.notes_path)
        return self

    def get_notes(self) -> List[Dict[str, str]]:
        if self._notes is None:
            self.load()
        return self._notes  # type: ignore[return-value]

    def run(self) -> pd.DataFrame:
        """
        Run extraction over all notes.

        Returns a DataFrame with columns:
          note_id, entity_text, label, start, end, negated
        """
        rows = []
        for note in self.get_notes():
            for ent in self.engine.extract(note["text"]):
                rows.append({
                    "note_id":     note["note_id"],
                    "entity_text": ent.text,
                    "label":       ent.label,
                    "start":       ent.start,
                    "end":         ent.end,
                    "negated":     ent.negated,
                })
        return pd.DataFrame(
            rows,
            columns=["note_id", "entity_text", "label", "start", "end", "negated"],
        )
