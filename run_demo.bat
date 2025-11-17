@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Ensure we are in the script directory
cd /d "%~dp0"

set DATA_DIR=annotation\sample_data
set MODEL_BASE=models\baseline_chargram
set MODEL_CAL=models\baseline_chargram_cal
set REPORT_BASE=reports\baseline_chargram
set REPORT_CAL=reports\baseline_chargram_cal

echo [1/12] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error

echo [2/12] Installing Python dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 goto :error

echo [3/12] Computing inter-annotator agreement on demo data...
python "annotation\adjudicate.py" iaa --input "%DATA_DIR%\demo_annotations.csv"
if errorlevel 1 goto :error

echo [4/12] Inserting adjudication stubs into the log (if any new)...
python "annotation\adjudicate.py" queue --input "%DATA_DIR%\demo_annotations.csv" --log "annotation\adjudication_log.md" --limit 20
if errorlevel 1 goto :error

echo [5/12] Finalizing dataset with majority fallback and exporting stratified splits...
python "annotation\finalize_dataset.py" combine --input "%DATA_DIR%\demo_annotations.csv" --log "annotation\adjudication_log.md" --output "%DATA_DIR%\final_dataset.csv" --strategy majority --export-splits
if errorlevel 1 goto :error

REM Optional: demonstrate group-aware splits if a groups mapping exists
if exist "%DATA_DIR%\groups.csv" (
  echo [5b/12] Regenerating splits with group-aware split (groups.csv present)...
  python "annotation\finalize_dataset.py" combine --input "%DATA_DIR%\demo_annotations.csv" --log "annotation\adjudication_log.md" --output "%DATA_DIR%\final_dataset.csv" --strategy majority --groups_csv "%DATA_DIR%\groups.csv" --group_column group --export-splits --group_aware_splits
  if errorlevel 1 goto :error
)

REM Sanity check that splits exist
if not exist "%DATA_DIR%\final_train.csv" (
  echo ERROR: Missing %DATA_DIR%\final_train.csv & goto :error
)
if not exist "%DATA_DIR%\final_val.csv" (
  echo ERROR: Missing %DATA_DIR%\final_val.csv & goto :error
)
if not exist "%DATA_DIR%\final_test.csv" (
  echo ERROR: Missing %DATA_DIR%\final_test.csv & goto :error
)

echo [6/12] Training baseline model (char n-gram TF-IDF + Logistic Regression)...
python "modeling\train_baseline.py" --input "%DATA_DIR%\final_dataset.csv" --use_splits --output_dir "%MODEL_BASE%"
if errorlevel 1 goto :error

REM Verify baseline artifacts exist before proceeding
if not exist "%MODEL_BASE%\vectorizer.pkl" (
  echo ERROR: Missing %MODEL_BASE%\vectorizer.pkl & goto :error
)
if not exist "%MODEL_BASE%\model.pkl" (
  echo ERROR: Missing %MODEL_BASE%\model.pkl & goto :error
)

echo [7/12] Single-text prediction (sanity check)...
python "modeling\predict.py" --model_dir "%MODEL_BASE%" --text "អរគុណច្រើន សេវាកម្មល្អបំផុត 🙄"
if errorlevel 1 goto :error

echo [8/12] Batch prediction and evaluation on test split...
python "modeling\predict.py" --model_dir "%MODEL_BASE%" --input_csv "%DATA_DIR%\final_test.csv" --output_csv "%DATA_DIR%\pred_test.csv"
if errorlevel 1 goto :error

echo [9/12] Error analysis and reports (baseline)...
python "modeling\error_analysis.py" --model_dir "%MODEL_BASE%" --input_csv "%DATA_DIR%\final_test.csv" --output_dir "%REPORT_BASE%"
if errorlevel 1 goto :error

REM Optional: demonstrate calibration and group-aware training if group column exists and validation split has rows
set "VAL_LINES="
for /f %%c in ('type "%DATA_DIR%\final_val.csv" ^| find /c /v ""') do set VAL_LINES=%%c
if not defined VAL_LINES set VAL_LINES=0
if %VAL_LINES% LEQ 1 (
  echo [10/12] Skipping calibrated model: validation split is empty or header-only.
) else (
  set "HAS_GROUP="
  for /f "usebackq delims=" %%g in (`python -c "import csv,sys,pathlib;p=pathlib.Path(r'%DATA_DIR%\final_train.csv');\nimport io\ntry:\n f=p.open('r',encoding='utf-8');\n r=csv.reader(f);\n h=next(r,[]);\n print('YES' if any((c or '').strip().lower()=='group' for c in h) else 'NO')\nexcept Exception:\n print('NO')"`) do set HAS_GROUP=%%g
  if /I "%HAS_GROUP%"=="YES" (
    echo [10/12] Training calibrated model with group-aware split awareness...
    python "modeling\train_baseline.py" --input "%DATA_DIR%\final_dataset.csv" --use_splits --output_dir "%MODEL_CAL%" --group_column group --calibrate platt
    if errorlevel 1 goto :error
  ) else (
    echo [10/12] Skipping calibrated model: group column not found in header.
  )
)

REM If calibrated model exists, run its analysis
if exist "%MODEL_CAL%\model.pkl" (
  echo [11/12] Error analysis and reports (calibrated, with reliability and group slices)...
  python "modeling\error_analysis.py" --model_dir "%MODEL_CAL%" --input_csv "%DATA_DIR%\final_test.csv" --output_dir "%REPORT_CAL%" --slice_column group --reliability_bins 15
  if errorlevel 1 goto :error
)

echo [12/12] Completed all steps.
echo.
echo Demo completed successfully.
goto :eof

:error
echo.
echo ERROR: Demo failed. See messages above for details.
exit /b 1
