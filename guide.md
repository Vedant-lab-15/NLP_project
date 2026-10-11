# MedExtract-Lite — Code Navigation & Developer Guide

> **Purpose of this file:** Walk through every source file, explain the key logic,
> and give you a step-by-step demo script so you can present or viva the project confidently.

---

## 1. How to Run the Project (Windows)

### Option A — One-Click (recommended)
Double-click **`run.bat`** in the project root.  
It handles everything automatically: install → generate data → test → launch app.

### Option B — Manual (step by step)

Open a terminal in the project folder and run:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic clinical notes
python data/generate_data.py

# 3. Run tests — must show 31 passed
pytest tests/test_ner.py -v

# 4. Launch the Streamlit web app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 2. File-by-File Guide

### `data/medical_dict.json`
**What it is:** The clinical lexicon — a JSON file with three lists.

```json
{
  "diseases": ["hypertension", "acute myocardial infarction", ...],
  "symptoms": ["chest pain", "shortness of breath", ...],
  "drugs":    ["aspirin", "metformin", ...]
}
```

- 30 diseases · 30 symptoms · 30 drugs = **90 terms total**
- Adding a new term is as simple as appending a string to the right list
- Terms are **case-insensitive** at matching time

---

### `data/generate_data.py`
**What it does:** Writes 5 synthetic discharge summaries to `clinical_notes.txt`.

- Each note is a plain-text block separated by `---`
- Structured like a real hospital discharge note (Admitting Diagnosis, History, Medications)
- Includes both affirmative and negated mentions for realistic testing

Run it:
```bash
python data/generate_data.py
```

---

### `src/nlp_engine.py` ⭐ Core file

This is the heart of the system. Read this file first.

#### Key classes and methods

| Symbol | Line | Purpose |
|---|---|---|
| `Entity` (dataclass) | top | Holds one extracted entity: text, label, start, end, negated |
| `ABBREV_RULES` | constant | List of `(regex_pattern, label)` for abbreviations |
| `_NEG_CUES` | constant | Pre-compiled regex for negation cues |
| `MedExtractEngine.__init__()` | method | Loads lexicon, sorts terms, compiles all patterns |
| `MedExtractEngine._lex_hits()` | method | Layer 1 — lexicon extraction |
| `MedExtractEngine._abbrev_hits()` | method | Layer 2 — regex abbreviation extraction |
| `MedExtractEngine._merge()` | static | Layer 3 — span deduplication |
| `MedExtractEngine._is_negated()` | static | Layer 4 — negation window check |
| `MedExtractEngine.extract()` | **main entry** | Runs all 4 layers, returns `List[Entity]` |
| `MedExtractEngine.summarise()` | method | Returns unique non-negated entities by label |

#### Walk-through of `extract(text)`

```
extract("No evidence of pneumonia. Patient has hypertension.")
    │
    ├─ _lex_hits(text)
    │   → finds "pneumonia" at [15:24] DISEASE
    │   → finds "hypertension" at [34:46] DISEASE
    │
    ├─ _abbrev_hits(text)
    │   → no abbreviation matches
    │
    ├─ _merge(lex_hits, abbrev_hits)
    │   → sorts by (start, -length), deduplicates overlaps
    │   → [(15, 24, "pneumonia", DISEASE), (34, 46, "hypertension", DISEASE)]
    │
    ├─ _is_negated(text, start=15)
    │   → looks back 60 chars before offset 15
    │   → finds "No evidence of" → negated=True
    │
    ├─ _is_negated(text, start=34)
    │   → looks back from offset 34
    │   → sentence boundary (.) found at offset 24 → window is " Patient has "
    │   → no cue found → negated=False
    │
    └─ Returns:
        [Entity("pneumonia",    DISEASE, 15, 24, negated=True),
         Entity("hypertension", DISEASE, 34, 46, negated=False)]
```

#### Why longest-match?

Terms are sorted by length (descending) before compilation:
```
"type 2 diabetes mellitus"  ← matched first (24 chars)
"diabetes mellitus"         ← skipped — span already occupied
"diabetes"                  ← skipped — span already occupied
```

This guarantees you get the most specific entity, not a fragment.

#### Why sentence-scoped negation?

The original window-only approach caused bleed-across:
```
"No diabetes mellitus. Patient has hypertension."
```
Without sentence scoping, "hypertension" would also be flagged negated because
"No" was within 60 chars. The fix: `_is_negated()` trims the look-back window
at the last `.`, `!`, `\n`, or `;`.

---

### `src/pipeline.py`

**What it does:** Loads `clinical_notes.txt` and builds a pandas DataFrame of results.

#### Key functions

| Symbol | Purpose |
|---|---|
| `load_notes(filepath)` | Reads the file, splits on `---`, returns `List[{note_id, text}]` |
| `ClinicalPipeline.load()` | Calls `load_notes()`, stores internally |
| `ClinicalPipeline.run()` | Iterates notes → `engine.extract()` → builds DataFrame |

#### DataFrame schema
```
note_id | entity_text | label   | start | end | negated
--------|-------------|---------|-------|-----|--------
001     | hypertension| DISEASE | 16    | 28  | False
001     | metformin   | DRUG    | 85    | 94  | False
```

---

### `tests/test_ner.py`

10 test classes, 31 assertions. Run with:
```bash
pytest tests/test_ner.py -v
```

| Class | What it tests |
|---|---|
| `TestInit` | Engine loads, lexicon has entries, missing file raises error |
| `TestDiseaseExtraction` | hypertension, diabetes, pneumonia detected |
| `TestSymptomExtraction` | chest pain, shortness of breath, fatigue |
| `TestDrugExtraction` | metformin, aspirin, atorvastatin |
| `TestNegation` | Negated entities flagged; excluded from `summarise()` |
| `TestAbbreviations` | COPD, HTN, DVT, UTI all map to DISEASE |
| `TestLongestMatch` | No overlapping spans; full phrase preferred over fragment |
| `TestEdgeCases` | Empty string, whitespace, non-medical text |
| `TestSummarise` | No duplicates; negated excluded; all three labels |
| `TestMultiEntity` | One note produces DISEASE + SYMPTOM + DRUG |

---

### `app.py`

Streamlit web application with 3 pages, light clinical theme.

| Function | Page | What it renders |
|---|---|---|
| `topbar()` | all | Deep-blue header bar with logo and badges |
| `sidebar()` | all | Navigation radio, entity legend pills, options |
| `page_extractor()` | Extractor | Text input + annotated output + KPI cards + entity table |
| `page_batch()` | Batch View | 5-note browser + bar chart + top-entity table + full results |
| `page_about()` | About | Architecture steps + entity type cards + stack table |

Helper functions:
- `annotate_html(text, entities)` — builds the highlighted HTML span output
- `kpi(label, value, sub, color)` — renders a metric card in HTML
- `pill(label)` — renders a coloured entity badge
- `to_df(entities)` — converts `List[Entity]` to a pandas DataFrame

---

### `run.bat` (Windows launcher)

Double-click to run. Does:
1. `pip install -r requirements.txt`
2. `python data/generate_data.py`
3. `pytest tests/test_ner.py -v`
4. `streamlit run app.py`

If any step fails, the window stays open and shows the error.

---

## 3. Demo Script (Examiner Walkthrough)

Follow this order when presenting:

### Step 1 — Launch
```
Double-click run.bat
→ Terminal installs deps, runs tests (31 passed), opens browser
```

### Step 2 — Extractor page
```
1. Show the pre-loaded demo note
2. Point out the structured sections (ADMITTING DIAGNOSIS, PAST MEDICAL HISTORY, etc.)
3. Click "Run Extraction"
4. Show the 4 KPI cards (Total / Diseases / Symptoms / Drugs)
5. Hover over a highlighted entity → tooltip shows label
6. Point to a struck-through entity (e.g. "atrial fibrillation") → "negated=True"
7. Show the entity table tabs (All / Diseases / Symptoms / Drugs)
8. Click "Export as CSV"
```

### Step 3 — Batch View page
```
1. Click sidebar → 📊 Batch View
2. Show 5-note KPI row
3. Use the dropdown to browse each note → annotation updates live
4. Point out the bar chart — DISEASE vs SYMPTOM vs DRUG
5. Show the "Most Frequent Entities" table
6. Use the label filter multiselect in "All Extracted Entities"
```

### Step 4 — About page
```
1. Walk through the 4 extraction layers (numbered cards)
2. Show the project structure code block
3. Explain the entity type cards and colour scheme
4. Point out the "Demo Mode Active" info box
```

---

## 4. Viva Q&A

**Q: Why rule-based instead of a transformer model like BERT?**  
A: The requirement is CPU-only with no heavy frameworks. Rule-based NLP is fast
(< 5ms per note), deterministic, fully explainable, and auditable — all properties
that matter in clinical settings where trust and transparency are critical.

**Q: How does longest-match work?**  
A: Terms in `_lexicon` are sorted by character length, descending. The first
pattern to match a span claims that character range. All subsequent patterns
skip it using the `occupied` interval list. This prevents "diabetes" from
shadowing "type 2 diabetes mellitus".

**Q: What happens when two entities overlap?**  
A: `_merge()` sorts combined hits by `(start, -length)` — longest span first on
ties. It walks the list keeping only spans that start at or after `last_end`.
Overlapping spans are dropped.

**Q: How reliable is the negation detection?**  
A: It handles the most common clinical patterns reliably. Sentence scoping
(cutting the window at `.`, `!`, `\n`) prevents negation bleed from one
sentence to the next. It can miss: indirect negation ("unlikely to be X"),
double negatives, and negation expressed more than 60 chars before the entity.

**Q: How would you extend the lexicon?**  
A: Open `data/medical_dict.json` and add a term to the appropriate list.
The engine re-sorts and re-compiles patterns on next load. No code changes needed.

**Q: How do you run just one test class?**  
```bash
pytest tests/test_ner.py::TestNegation -v
```

**Q: What does `negated=True` mean in the output?**  
A: The entity was detected but is under the scope of a negation cue in the
preceding sentence. `summarise()` excludes these from the clean output so
downstream consumers only see confirmed clinical findings.
