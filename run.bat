@echo off
title MedExtract-Lite Launcher
color 1F

echo.
echo  ============================================================
echo   MedExtract-Lite : Clinical NER System
echo   Rule + Dictionary + Regex Based NLP (CPU Only)
echo  ============================================================
echo.

:: ── Step 1: Install dependencies ────────────────────────────────────────────
echo  [1/4]  Installing Python dependencies...
echo  ─────────────────────────────────────────
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo  [ERROR] pip install failed.
    echo  Make sure Python and pip are installed and in your PATH.
    pause
    exit /b 1
)
echo  [OK] Dependencies installed.
echo.

:: ── Step 2: Generate sample data ────────────────────────────────────────────
echo  [2/4]  Generating synthetic clinical notes...
echo  ─────────────────────────────────────────────
python data/generate_data.py
if errorlevel 1 (
    echo.
    echo  [WARNING] Data generation failed. App will use built-in demo data.
    echo.
) else (
    echo  [OK] Clinical notes generated.
    echo.
)

:: ── Step 3: Run tests ────────────────────────────────────────────────────────
echo  [3/4]  Running test suite...
echo  ──────────────────────────────
pytest tests/test_ner.py -v --tb=short
if errorlevel 1 (
    echo.
    echo  [WARNING] Some tests failed. Review output above.
    echo  The app will still launch.
    echo.
) else (
    echo.
    echo  [OK] All 31 tests passed.
    echo.
)

:: ── Step 4: Launch app ───────────────────────────────────────────────────────
echo  [4/4]  Launching Streamlit web application...
echo  ───────────────────────────────────────────────
echo.
echo  ============================================================
echo   App running at:  http://localhost:8501
echo   Press Ctrl+C in this window to stop the server.
echo  ============================================================
echo.

streamlit run app.py --server.headless true

echo.
echo  Server stopped.
pause
