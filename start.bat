@echo off
title AI Video Object Remover Web Studio
echo ========================================================
echo    Starting AI Video Object Remover Web Studio
echo ========================================================
echo Installing / checking requirements...
python -m pip install -r backend\requirements.txt
echo Launching application...
python run.py
pause
