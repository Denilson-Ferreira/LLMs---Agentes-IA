@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"

set "PROJECT=%~dp0"
set "PYTHON=%PROJECT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" set "PYTHON=%PROJECT%..\..\..\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Ambiente virtual nao encontrado. Consulte o README.md para instalar as dependencias.
    exit /b 1
)

pushd "%PROJECT%"
"%PYTHON%" main.py
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
