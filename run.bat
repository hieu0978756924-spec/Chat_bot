@echo off
REM ======================================================================
REM  Tro ly dat ban nha hang (Local LLM) - khoi dong tren Windows
REM  File nay PHAI dung xuong dong CRLF va chi ky tu ASCII (cmd.exe doc
REM  sai file UTF-8/LF sau "chcp 65001").
REM  Tham so truyen tiep cho main.py, vd: run.bat --debug
REM ======================================================================
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
title Tro ly dat ban nha hang
cd /d "%~dp0"

echo ======================================================================
echo   TRO LY DAT BAN NHA HANG - LOCAL LLM
echo ======================================================================

REM ---- 1. Python / venv ------------------------------------------------
set "PY=venv\Scripts\python.exe"
if not exist "%PY%" (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [LOI] Khong tim thay Python. Cai Python 3.10+ tu python.org roi chay lai.
        goto :end
    )
    echo [!] Chua co moi truong ao venv.
    set "ANS="
    set /p "ANS=Tao venv va cai thu vien tu requirements.txt? (y/n): "
    if /i not "!ANS!"=="y" goto :end
    python -m venv venv || goto :end
    "%PY%" -m pip install -r requirements.txt || goto :end
)
echo [1/4] Python: OK

REM ---- 2. Ollama + model -----------------------------------------------
set "MODEL=qwen2.5:3b"
for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    if /i "%%a"=="OLLAMA_MODEL" set "MODEL=%%b"
)
where ollama >nul 2>&1
if errorlevel 1 (
    echo [2/4] [!] Khong thay lenh 'ollama'. Chatbot se dung NLU du phong ^(kem chinh xac hon^).
) else (
    ollama list >nul 2>&1
    if errorlevel 1 (
        echo [2/4] [!] Ollama chua chay. Mo ung dung Ollama hoac chay 'ollama serve'.
    ) else (
        ollama list | findstr /i /c:"!MODEL!" >nul 2>&1
        if errorlevel 1 (
            echo [2/4] [!] Chua co model !MODEL! trong Ollama.
            set "ANS="
            set /p "ANS=Tai model !MODEL! ngay bay gio? (y/n): "
            if /i "!ANS!"=="y" ollama pull !MODEL!
        ) else (
            echo [2/4] Ollama: model !MODEL! san sang
        )
    )
)

REM ---- 3. CSDL + FAISS index -------------------------------------------
echo [3/4] Kiem tra CSDL va chi muc chinh sach...
"%PY%" scripts\init_db.py --if-outdated --index
if errorlevel 1 (
    echo [LOI] Khoi tao CSDL / chi muc that bai. Xem thong bao o tren.
    goto :end
)

REM ---- 4. Chay chatbot -------------------------------------------------
echo [4/4] Khoi dong tro ly...
echo ----------------------------------------------------------------------
"%PY%" main.py %*

:end
echo.
pause
endlocal
