@echo off
REM Chay giao dien web TableNow tai http://127.0.0.1:8000 (giu CRLF + ASCII cho file nay)
setlocal EnableExtensions
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
title TableNow - giao dien web
cd /d "%~dp0"
set "PY=venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [LOI] Chua co venv. Chay run.bat mot lan de cai dat truoc.
    goto :end
)
"%PY%" scripts\init_db.py --if-outdated --index
if errorlevel 1 goto :end
"%PY%" web\server.py %*
:end
echo.
pause
endlocal
