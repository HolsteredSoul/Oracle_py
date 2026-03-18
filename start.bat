@echo off
title Oracle Paper Trading
cd /d "%~dp0"

echo Starting Oracle agent (betfair-paper mode)...
start "Oracle Agent" cmd /k python main.py --betfair-paper

echo Starting Oracle dashboard...
start "Oracle Dashboard" cmd /k python -m streamlit run src/dashboard/app.py

echo.
echo Both processes started.
echo   Agent:     logs in its own window
echo   Dashboard: http://localhost:8501
echo.
echo Close the terminal windows to stop.
