"""
test_ner.py
-----------
PyTest suite for MedExtract-Lite.

Run from project root:
    pytest tests/test_ner.py -v
"""

import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.nlp_engine import MedExtractEngine, Entity

DICT_PATH = os.path.join(ROOT, "data", "medical_dict.json")


@pytest.fixture(scope="module")
def engine():
    return MedExtractEngine(dict_path=DICT_PATH)


# ── T1: Engine initialisation ────────────────────────────────────────────────

class TestInit:
    def test_lexicon_populated(self, engine):
        assert len(engine._lexicon) >= 30

    def test_sorted_terms_longest_first(self, engine):
        terms = engine._sorted_terms
        for i in range(len(terms) - 1):
            assert len(terms[i]) >= len(terms[i + 1])

    def test_missing_dict_raises(self):
        with pytest.raises(FileNotFoundError):
            MedExtractEngine(dict_path="/no/such/path.json")

    def test_inline_dict_works(self):
        eng = MedExtractEngine(medical_dict={
            "diseases": ["hypertension"],
            "symptoms": ["headache"],
            "drugs":    ["aspirin"],
        })
        entities = eng.extract("Hypertension with headache. Prescribed aspirin.")
        labels = {e.label for e in entities}
        assert labels == {"DISEASE", "SYMPTOM", "DRUG"}


# ── T2: Disease extraction ───────────────────────────────────────────────────

class TestDiseaseExtraction:
    def test_hypertension(self, engine):
        ents = engine.extract("Patient has hypertension.")
        assert any(e.label == "DISEASE" and "hypertension" in e.text.lower()
                   for e in ents)

    def test_diabetes(self, engine):
        ents = engine.extract("Known type 2 diabetes mellitus.")
        assert any("diabetes" in e.text.lower() for e in ents if e.label == "DISEASE")

    def test_pneumonia(self, engine):
        ents = engine.extract("Admitted with pneumonia.")
        assert any("pneumonia" in e.text.lower() for e in ents if e.label == "DISEASE")


# ── T3: Symptom extraction ───────────────────────────────────────────────────

class TestSymptomExtraction:
    def test_chest_pain(self, engine):
        ents = engine.extract("Complains of chest pain.")
        assert any(e.label == "SYMPTOM" and "chest pain" in e.text.lower()
                   for e in ents)

    def test_shortness_of_breath(self, engine):
        ents = engine.extract("Reports shortness of breath.")
        assert any("breath" in e.text.lower() for e in ents if e.label == "SYMPTOM")

    def test_fatigue(self, engine):
        ents = engine.extract("The patient reports fatigue.")
        assert any(e.label == "SYMPTOM" and "fatigue" in e.text.lower()
                   for e in ents)


# ── T4: Drug extraction ──────────────────────────────────────────────────────

class TestDrugExtraction:
    def test_metformin(self, engine):
        ents = engine.extract("Prescribed metformin 500mg.")
        assert any(e.label == "DRUG" and "metformin" in e.text.lower()
                   for e in ents)

    def test_aspirin(self, engine):
        ents = engine.extract("Started on aspirin 75mg.")
        assert any(e.label == "DRUG" and "aspirin" in e.text.lower()
                   for e in ents)

    def test_atorvastatin(self, engine):
        ents = engine.extract("Atorvastatin 40mg at night.")
        assert any(e.label == "DRUG" and "atorvastatin" in e.text.lower()
                   for e in ents)


# ── T5: Negation detection ───────────────────────────────────────────────────

class TestNegation:
    def test_no_evidence_of(self, engine):
        ents = engine.extract("No evidence of pneumonia.")
        disease_ents = [e for e in ents if e.label == "DISEASE"]
        assert any(e.negated for e in disease_ents), \
            f"Expected negated entity, got: {disease_ents}"

    def test_denied_symptom(self, engine):
        ents = engine.extract("Patient denies chest pain.")
        symptom_ents = [e for e in ents if e.label == "SYMPTOM"]
        assert any(e.negated for e in symptom_ents)

    def test_affirmative_not_negated(self, engine):
        ents = engine.extract("Diagnosed with hypertension.")
        disease_ents = [e for e in ents if e.label == "DISEASE"]
        assert any(not e.negated for e in disease_ents)

    def test_negated_excluded_from_summary(self, engine):
        ents = engine.extract("No history of diabetes mellitus.")
        summary = engine.summarise(ents)
        assert not any("diabetes" in t.lower() for t in summary["DISEASE"])


# ── T6: Abbreviation rules ───────────────────────────────────────────────────

class TestAbbreviations:
    def test_copd(self, engine):
        ents = engine.extract("History of COPD exacerbation.")
        assert any("COPD" in e.text for e in ents if e.label == "DISEASE")

    def test_htn(self, engine):
        ents = engine.extract("Background of HTN.")
        assert any("HTN" in e.text for e in ents if e.label == "DISEASE")

    def test_dvt(self, engine):
        ents = engine.extract("DVT confirmed on Doppler.")
        assert any("DVT" in e.text for e in ents if e.label == "DISEASE")

    def test_uti(self, engine):
        ents = engine.extract("Consistent with UTI.")
        assert any("UTI" in e.text for e in ents if e.label == "DISEASE")


# ── T7: Longest-match ────────────────────────────────────────────────────────

class TestLongestMatch:
    def test_no_overlapping_spans(self, engine):
        ents = engine.extract(
            "Congestive heart failure secondary to acute myocardial infarction."
        )
        spans = sorted((e.start, e.end) for e in ents)
        for i in range(len(spans) - 1):
            assert spans[i][1] <= spans[i + 1][0], \
                f"Overlapping spans: {spans[i]} and {spans[i+1]}"

    def test_full_phrase_preferred(self, engine):
        ents = engine.extract("type 2 diabetes mellitus")
        texts = [e.text.lower() for e in ents if e.label == "DISEASE"]
        assert any("type 2" in t for t in texts)


# ── T8: Edge cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string(self, engine):
        assert engine.extract("") == []

    def test_whitespace_only(self, engine):
        assert engine.extract("   \n\t  ") == []

    def test_no_medical_content(self, engine):
        result = engine.extract("The weather is sunny today.")
        assert isinstance(result, list)

    def test_uppercase_input(self, engine):
        ents = engine.extract("HYPERTENSION AND FATIGUE NOTED.")
        labels = {e.label for e in ents}
        assert "DISEASE" in labels or "SYMPTOM" in labels


# ── T9: Summarise ────────────────────────────────────────────────────────────

class TestSummarise:
    def test_no_duplicates(self, engine):
        ents = engine.extract("Hypertension noted. Hypertension documented again.")
        summary = engine.summarise(ents)
        diseases = [d.lower() for d in summary["DISEASE"]]
        assert diseases.count("hypertension") == 1

    def test_negated_excluded(self, engine):
        ents = engine.extract("No diabetes mellitus. Patient has hypertension.")
        summary = engine.summarise(ents)
        assert not any("diabetes" in t.lower() for t in summary["DISEASE"])
        assert any("hypertension" in t.lower() for t in summary["DISEASE"])

    def test_all_labels_present(self, engine):
        note = (
            "Patient has hypertension. Complains of chest pain. "
            "Prescribed aspirin."
        )
        ents = engine.extract(note)
        summary = engine.summarise(ents)
        assert summary["DISEASE"]
        assert summary["SYMPTOM"]
        assert summary["DRUG"]


# ── T10: Multi-entity note ───────────────────────────────────────────────────

class TestMultiEntity:
    NOTE = (
        "Admitted with acute myocardial infarction. "
        "Presented with chest pain and diaphoresis. "
        "Started on aspirin, atorvastatin, and metoprolol."
    )

    def test_all_three_labels(self, engine):
        ents = engine.extract(self.NOTE)
        labels = {e.label for e in ents}
        assert "DISEASE" in labels
        assert "SYMPTOM" in labels
        assert "DRUG" in labels
