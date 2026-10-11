"""
nlp_engine.py
-------------
MedExtract-Lite: Lightweight rule + regex + lexicon NER engine.

No PyTorch. No Hugging Face. Pure Python 3.8+.

Extraction layers:
  1. Dictionary / lexicon matching  (longest-match, case-insensitive)
  2. Regex abbreviation rules       (COPD, HTN, DVT, UTI, ...)
  3. Window-based negation tagging  (no, denied, without, ...)

Entity labels: DISEASE | SYMPTOM | DRUG
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    text:    str
    label:   str          # DISEASE | SYMPTOM | DRUG
    start:   int
    end:     int
    negated: bool = False

    def to_dict(self) -> dict:
        return {
            "text":    self.text,
            "label":   self.label,
            "start":   self.start,
            "end":     self.end,
            "negated": self.negated,
        }


# ---------------------------------------------------------------------------
# Abbreviation / regex rules
# ---------------------------------------------------------------------------

ABBREV_RULES: List[Tuple[str, str]] = [
    (r"\bCOPD\b",                                        "DISEASE"),
    (r"\bHTN\b",                                         "DISEASE"),
    (r"\bDVT\b",                                         "DISEASE"),
    (r"\bUTI\b",                                         "DISEASE"),
    (r"\bCAD\b",                                         "DISEASE"),
    (r"\bCHF\b",                                         "DISEASE"),
    (r"\bAFib\b",                                        "DISEASE"),
    (r"\bA-?[Ff]ib(?:rillation)?\b",                    "DISEASE"),
    (r"\bAKI\b",                                         "DISEASE"),
    (r"\bCKD\b",                                         "DISEASE"),
    (r"\bTIA\b",                                         "DISEASE"),
    (r"\bPE\b",                                          "DISEASE"),
    (r"\btype\s+[12]\s+diabetes(?:\s+mellitus)?\b",     "DISEASE"),
    (r"\bCOVID-?19\b",                                   "DISEASE"),
    (r"\bshortness\s+of\s+breath\b",                    "SYMPTOM"),
    (r"\bchest\s+(?:pain|tightness|pressure)\b",        "SYMPTOM"),
    (r"\bweight\s+(?:loss|gain)\b",                     "SYMPTOM"),
    (r"\bloss\s+of\s+(?:appetite|consciousness)\b",     "SYMPTOM"),
    (r"\bpedal\s+edema\b",                              "SYMPTOM"),
    (r"\bproductive\s+cough\b",                         "SYMPTOM"),
    (r"\baltered\s+mental\s+status\b",                  "SYMPTOM"),
]

# Negation cues (look-back window)
_NEG_CUES = re.compile(
    r"\b(?:no|not|never|denies|denied|without|absence\s+of|"
    r"no\s+evidence\s+of|no\s+history\s+of|ruled?\s+out)\b",
    re.IGNORECASE,
)
_NEG_WINDOW = 60   # characters to look back


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MedExtractEngine:
    """
    Lightweight biomedical NER engine.

    Parameters
    ----------
    dict_path    : path to medical_dict.json (used when medical_dict is None)
    medical_dict : pre-loaded dict {"diseases": [...], "symptoms": [...], "drugs": [...]}
    """

    # Built-in compact demo dictionary — used when no file is found
    _DEMO_DICT: Dict[str, List[str]] = {
        "diseases": [
            "hypertension", "diabetes mellitus", "type 2 diabetes mellitus",
            "acute myocardial infarction", "heart failure",
            "pneumonia", "sepsis", "stroke", "asthma", "anemia",
        ],
        "symptoms": [
            "chest pain", "shortness of breath", "fatigue", "fever",
            "nausea", "headache", "dizziness", "cough", "diaphoresis",
            "pedal edema", "leg swelling", "dyspnea",
        ],
        "drugs": [
            "aspirin", "metformin", "atorvastatin", "lisinopril",
            "metoprolol", "furosemide", "warfarin", "clopidogrel",
            "albuterol", "prednisone",
        ],
    }

    def __init__(
        self,
        dict_path:    Optional[str]                  = None,
        medical_dict: Optional[Dict[str, List[str]]] = None,
    ):
        if medical_dict is not None:
            raw = medical_dict
        elif dict_path and os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        else:
            if dict_path and not os.path.exists(dict_path):
                raise FileNotFoundError(
                    f"Dictionary not found: {dict_path}\n"
                    "Run: python data/generate_data.py"
                )
            # Fallback to demo dict (no file path given)
            raw = self._DEMO_DICT

        self.medical_dict = raw
        self._lexicon: Dict[str, str] = {}   # normalised_term -> label

        label_map = {"diseases": "DISEASE", "symptoms": "SYMPTOM", "drugs": "DRUG"}
        for category, label in label_map.items():
            for term in raw.get(category, []):
                self._lexicon[term.lower().strip()] = label

        # Longest terms first — prevents sub-term shadowing
        self._sorted_terms = sorted(
            self._lexicon.keys(), key=len, reverse=True
        )
        # Pre-compile one pattern per lexicon entry
        self._lex_patterns: List[Tuple[re.Pattern, str, str]] = [
            (
                re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE),
                t,
                self._lexicon[t],
            )
            for t in self._sorted_terms
        ]
        # Pre-compile abbreviation rules
        self._abbrev_patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(p, re.IGNORECASE), lbl)
            for p, lbl in ABBREV_RULES
        ]

    # ------------------------------------------------------------------
    # Internal: lexicon hits
    # ------------------------------------------------------------------

    def _lex_hits(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Return (start, end, surface, label) for all lexicon matches."""
        hits: List[Tuple[int, int, str, str]] = []
        occupied: List[Tuple[int, int]] = []   # sorted non-overlapping spans

        for pat, term, label in self._lex_patterns:
            for m in pat.finditer(text):
                s, e = m.start(), m.end()
                # Skip if this span overlaps an already-claimed span
                if any(os_ < e and oe > s for os_, oe in occupied):
                    continue
                occupied.append((s, e))
                occupied.sort()
                hits.append((s, e, m.group(0), label))

        return hits

    # ------------------------------------------------------------------
    # Internal: abbreviation hits
    # ------------------------------------------------------------------

    def _abbrev_hits(self, text: str) -> List[Tuple[int, int, str, str]]:
        hits: List[Tuple[int, int, str, str]] = []
        for pat, label in self._abbrev_patterns:
            for m in pat.finditer(text):
                hits.append((m.start(), m.end(), m.group(0), label))
        return hits

    # ------------------------------------------------------------------
    # Internal: merge and deduplicate spans
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(
        lex: List[Tuple[int, int, str, str]],
        abr: List[Tuple[int, int, str, str]],
    ) -> List[Tuple[int, int, str, str]]:
        """Combine hits, keep longest span on overlap, sort by offset."""
        combined = lex + abr
        combined.sort(key=lambda h: (h[0], -(h[1] - h[0])))
        merged: List[Tuple[int, int, str, str]] = []
        last_end = -1
        for s, e, surf, lbl in combined:
            if s >= last_end:
                merged.append((s, e, surf, lbl))
                last_end = e
        return merged

    # ------------------------------------------------------------------
    # Internal: negation check
    # ------------------------------------------------------------------

    @staticmethod
    def _is_negated(text: str, start: int) -> bool:
        """
        True if a negation cue exists within _NEG_WINDOW chars before start,
        but does NOT cross a sentence boundary (period, exclamation, newline).
        """
        raw_window = text[max(0, start - _NEG_WINDOW): start]
        # Only use the portion after the last sentence boundary
        boundary = max(
            raw_window.rfind("."),
            raw_window.rfind("!"),
            raw_window.rfind("\n"),
            raw_window.rfind(";"),
        )
        window = raw_window[boundary + 1:] if boundary >= 0 else raw_window
        return bool(_NEG_CUES.search(window))

    # ------------------------------------------------------------------
    # Public: extract
    # ------------------------------------------------------------------

    def extract(self, text: str) -> List[Entity]:
        """
        Extract biomedical entities from a clinical text string.

        Returns a list of Entity objects sorted by character offset.
        """
        if not text or not text.strip():
            return []

        lex_hits = self._lex_hits(text)
        abr_hits = self._abbrev_hits(text)
        all_hits = self._merge(lex_hits, abr_hits)

        entities: List[Entity] = []
        for s, e, surf, lbl in all_hits:
            entities.append(Entity(
                text    = surf,
                label   = lbl,
                start   = s,
                end     = e,
                negated = self._is_negated(text, s),
            ))

        return entities

    # ------------------------------------------------------------------
    # Public: summarise
    # ------------------------------------------------------------------

    def summarise(self, entities: List[Entity]) -> Dict[str, List[str]]:
        """Return unique non-negated entity texts grouped by label."""
        out: Dict[str, List[str]] = {"DISEASE": [], "SYMPTOM": [], "DRUG": []}
        seen: set = set()
        for e in entities:
            if e.negated:
                continue
            key = (e.label, e.text.lower())
            if key not in seen:
                seen.add(key)
                out[e.label].append(e.text)
        return out
