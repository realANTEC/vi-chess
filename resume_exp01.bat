@echo off
REM Launched by Windows Task Scheduler — runs the resumable experiment.
REM Output appends to exp01_run_detached.log. Skips matchups whose JSON checkpoints exist.
"C:\Users\vansh\AppData\Roaming\Python\Python312\Scripts\uv.exe" --directory C:\Users\vansh\VI_CHESS run python -u -m experiments.run_exp01 >> C:\Users\vansh\VI_CHESS\exp01_run_detached.log 2>&1
