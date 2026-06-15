@echo off
REM Example end-to-end run on the EED reference stream.
setlocal
set PY=C:\ProgramData\anaconda3\python.exe
if not exist "%PY%" set PY=python

set INPUT=..\Data\EED\what_is_background\events_filtered.txt
set OUT=out

"%PY%" "%~dp0segment.py" --input "%INPUT%" --out "%OUT%" --render-frames
if errorlevel 1 goto :end
"%PY%" "%~dp0visualize.py" --out "%OUT%"
:end
endlocal
