@echo off
chcp 65001 > nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo 先に install.bat を実行してください。
    pause
    exit /b 1
)
echo 起動中... 初回はモデルのダウンロードに数分かかります。
echo 起動後、ブラウザで http://localhost:8510 が開きます。終了するにはこの画面で Ctrl+C
start "" cmd /c "timeout /t 15 > nul && start http://localhost:8510"
".venv\Scripts\python.exe" -m uvicorn app:app --port 8510 --env-file .env
pause
