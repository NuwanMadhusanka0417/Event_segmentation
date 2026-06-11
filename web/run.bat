@echo off
setlocal
set PY="C:\ProgramData\anaconda3\python.exe"
if not exist %PY% set PY=python
%PY% "%~dp0app.py"
endlocal
