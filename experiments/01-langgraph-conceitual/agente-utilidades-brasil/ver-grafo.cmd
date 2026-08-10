@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"

set "PROJECT=%~dp0"
set "PYTHON=%PROJECT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" set "PYTHON=%PROJECT%..\..\..\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Ambiente virtual nao encontrado. Consulte o README.md para instalar as dependencias.
    pause
    exit /b 1
)

pushd "%PROJECT%"
"%PYTHON%" visualizar_grafo.py --abrir
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
