@echo off
REM Launched by Windows Task Scheduler ("ViChessPhase3"). Trains the learned
REM aggregator if its pickle is missing, then runs the Phase 3 experiment.
REM All output appends to phase3_run.log.

cd /d C:\Users\vansh\VI_CHESS

if not exist "experiments\models\phase3_mlp.pkl" (
    echo Training learned aggregator >> phase3_run.log 2>&1
    "C:\Users\vansh\AppData\Roaming\Python\Python312\Scripts\uv.exe" run python -u -m vi_chess.training.train >> phase3_run.log 2>&1
    if errorlevel 1 (
        echo Training failed, aborting >> phase3_run.log
        exit /b 1
    )
)

echo Starting Phase 3 experiment >> phase3_run.log 2>&1
"C:\Users\vansh\AppData\Roaming\Python\Python312\Scripts\uv.exe" run python -u -m experiments.run_phase3 >> phase3_run.log 2>&1
