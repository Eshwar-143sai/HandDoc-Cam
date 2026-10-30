@echo off
cd /d "%~dp0"
title HandDoc Cam Live Preview

echo ========================================================
echo               HandDoc Cam - Live Preview
echo ========================================================
echo.
echo Camera: Initializing Camera 0 (webcam)...
echo Object: sample_phone.png (13 cm phone)
echo.
echo Hotkeys while the camera window is focused:
echo   o       : Open any photo file dialog
echo   s       : Set/type real object size (cm or 'phone', 'laptop')
echo   g       : Toggle Size Gate ON / OFF (hold oversized objects)
echo   p       : Cycle FPS HUD mode (compact -> full -> off)
echo   + / -   : Zoom / scale object in your hands
echo   c       : Calibrate hand span with credit card
echo   q / Esc : Quit preview
echo ========================================================
echo.

set PYTHON_EXE="C:\Users\saies\AppData\Local\Programs\Python\Python311\python.exe"
if not exist %PYTHON_EXE% (
    set PYTHON_EXE=python
)

%PYTHON_EXE% handdoc_cam.py sample_phone.png --no-vcam

echo.
echo ========================================================
echo HandDoc Cam session ended.
echo ========================================================
pause
