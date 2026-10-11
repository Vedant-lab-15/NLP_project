# MedExtract-Lite
### Rule & Dictionary-Based Clinical NER System

> Lightweight, CPU-only biomedical Named Entity Recognition.  
> Extracts **DISEASE**, **SYMPTOM**, and **DRUG** entities from clinical discharge summaries  
> using pure Python — no PyTorch, no Hugging Face, no GPU required.

---

## Quick Start (Windows — One Click)

Double-click **`run.bat`** in the project folder.

It will automatically:
1. Install all Python dependencies
2. Generate sample clinical notes
3. Run the test suite
4. Launch the web app at **http://localhost:8501**

---

## Manual Setup

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate sample data *(optional — app works without this)*
```bash
python data/generate_data.py
```

### Step 3 — Run the test suite
```bash
pytest tests/test_ner.py -v
```
Expected output: **31 passed** in under 1 second.

### Step 4 — Launch the web app
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## Project Structure

```
clinical_ner_simple/
│
├── data/
│   ├── generate_data.py        ← Generates 5 synthetic clinical notes
│   ├── clinical_notes.txt      ← Auto-generated discharge summaries
│   └── medical_dict.json       ← 90-term clinical lexicon (diseases/symptoms/drugs)
│
├── src/
│   ├── __init__.py
│   ├── nlp_engine.py           ← Core NER engine (lexicon + regex + negation)
│   └── pipeline.py             ← Note loader & pandas DataFrame builder
│
├── tests/
│   └── test_ner.py             ← 31-assertion PyTest suite (10 test classes)
│
├── app.py                      ← Streamlit web application (3 pages)
├── run.bat                     ← Windows one-click launcher ⭐
├── run.sh                      ← Linux / macOS launcher
├── guide.md                    ← Code navigation & viva guide
├── README.md                   ← This file
└── requirements.txt            ← Python package dependencies
```

---

## Application Pages

| Page | Description |
|---|---|
| 🔬 **Extractor** | Paste any clinical text — entities are extracted and highlighted in real time |
| 📊 **Batch View** | Browse 5 built-in demo notes with per-note annotation and statistics charts |
| ℹ️ **About** | Architecture overview, technology stack, and run commands |

---

## How It Works — Extraction Pipeline

```
Input Clinical Text
        │
        ▼
┌─────────────────────────────────┐
│  Layer 1: Lexicon Matching      │  Looks up 90 terms from medical_dict.json
│                                 │  Longest-match: "type 2 diabetes mellitus"
│                                 │  matched before "diabetes mellitus"
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Layer 2: Regex Abbreviations   │  COPD · HTN · DVT · UTI · CAD · CHF
│                                 │  AKI · TIA · COVID-19 · type 2 DM …
│                                 │  Also: "shortness of breath", "pedal edema"
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Layer 3: Span Deduplication    │  Longest span wins on overlap
│                                 │  Zero redundant entities guaranteed
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Layer 4: Negation Detection    │  Sentence-scoped 60-char look-back
│                                 │  Cues: no · denied · without ·
│                                 │  no evidence of · no history of …
└────────────────┬────────────────┘
                 │
                 ▼
          List[Entity]
    (text, label, start, end, negated)
                 │
                 ▼
    Streamlit UI · DataFrame · CSV Export
```

---

## Entity Types

| Entity | Colour | Examples |
|---|---|---|
| 🦠 **DISEASE** | Red | hypertension, acute myocardial infarction, COPD |
| 🔍 **SYMPTOM** | Amber | chest pain, shortness of breath, pedal edema |
| 💊 **DRUG** | Green | aspirin, metformin, atorvastatin |

Negated entities (e.g. *"no evidence of pneumonia"*) are flagged with `negated=True`
and shown with strikethrough in the UI.

---

## Python API

```python
from src.nlp_engine import MedExtractEngine

# Load from dictionary file
engine = MedExtractEngine(dict_path="data/medical_dict.json")

note = """
ADMITTING DIAGNOSIS:
Patient admitted with acute myocardial infarction.
Presented with chest pain and diaphoresis.
No history of atrial fibrillation.
Prescribed aspirin and atorvastatin on discharge.
"""

# Extract entities
entities = engine.extract(note)

for e in entities:
    status = "NEGATED" if e.negated else "confirmed"
    print(f"[{e.label:7s}] {e.text:<35s} ({status})")

# Get a clean summary (negated entities excluded)
summary = engine.summarise(entities)
print(summary)
# {'DISEASE': ['acute myocardial infarction'],
#  'SYMPTOM': ['chest pain', 'diaphoresis'],
#  'DRUG':    ['aspirin', 'atorvastatin']}
```

### Pipeline API (batch processing)

```python
from src.nlp_engine import MedExtractEngine
from src.pipeline import ClinicalPipeline

engine   = MedExtractEngine(dict_path="data/medical_dict.json")
pipeline = ClinicalPipeline(engine=engine, notes_path="data/clinical_notes.txt")

df = pipeline.run()   # returns pandas DataFrame
print(df.head())
#   note_id          entity_text    label  start  end  negated
# 0     001          hypertension  DISEASE     16   28    False
# 1     001  type 2 diabetes mel…  DISEASE     31   56    False
```

---

## Running Tests

```bash
# Run all tests with verbose output
pytest tests/test_ner.py -v

# Run a specific test class
pytest tests/test_ner.py::TestNegation -v

# Quick pass/fail summary
pytest tests/test_ner.py -q
```

### Test Coverage

| Class | Tests | What it covers |
|---|---|---|
| `TestInit` | 4 | Engine init, lexicon loading, error handling |
| `TestDiseaseExtraction` | 3 | Disease entity detection |
| `TestSymptomExtraction` | 3 | Symptom entity detection |
| `TestDrugExtraction` | 3 | Drug / medication detection |
| `TestNegation` | 4 | Negation flagging + summary exclusion |
| `TestAbbreviations` | 4 | COPD, HTN, DVT, UTI abbreviations |
| `TestLongestMatch` | 2 | No overlapping spans + full phrase preferred |
| `TestEdgeCases` | 4 | Empty input, whitespace, no medical content |
| `TestSummarise` | 3 | Deduplication, negation exclusion |
| `TestMultiEntity` | 1 | All three labels in one note |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0.0 | DataFrame construction and analytics |
| `streamlit` | ≥ 1.35.0 | Web application framework |
| `pytest` | ≥ 8.0.0 | Unit testing |

**No ML frameworks.** All NLP logic uses Python's built-in `re` and `json` modules.

---

## Demo Mode

The app works **without running any setup commands** first.

- If `medical_dict.json` is missing, the engine uses a built-in compact dictionary.
- The Batch View page uses 5 hardcoded demo notes — no file I/O needed.
- Run `python data/generate_data.py` only if you want the full notes file.

---

## Limitations

- Lexicon-based matching does not handle spelling variants or typos.
- Negation is window-based (60 chars, sentence-scoped) — complex negation structures may be missed.
- No coreference resolution (pronouns are not linked to their antecedents).
- Extending the lexicon: simply add terms to the appropriate list in `data/medical_dict.json`.

---

## License

MIT — free to use, modify, and distribute for educational and research purposes.
