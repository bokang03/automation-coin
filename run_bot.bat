@echo off
cd /d "%~dp0"
.venv\Scripts\python -m aoabot.bot %*
