@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo === Installing Reprojection toolbox===

REM --- Config ---
set PY_VERSION=3.12.10
set PY_MAJOR=312
set PY_URL=https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-embed-amd64.zip
set GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py

set ROOT_DIR=%LOCALAPPDATA%\Reprojection_toolbox
set PYTHON_DIR=%ROOT_DIR%\python
set PY_ZIP=%ROOT_DIR%\python_embed.zip
set PYTHON_EXE=%PYTHON_DIR%\python.exe
set GET_PIP_FILE=%ROOT_DIR%\get-pip.py

cd /d "%ROOT_DIR%"

REM --- Extract repo from ZIP ---
set "ZIP_PATH=%~dp0\RT.zip"
set "METASHAPE_PATH=%~dp0\metashape-2.3.0-cp39.cp310.cp311.cp312.cp313-none-win_amd64.whl"
set "DEST_DIR=%ROOT_DIR%"
set "RT_DIR=%ROOT_DIR%\RT"

echo Extracting archive...
echo Extracting RT.zip...
powershell -NoLogo -NoProfile -Command ^
	"Expand-Archive -Path '%ZIP_PATH%' -DestinationPath '%DEST_DIR%' -Force"



REM --- Download Python embeddable if missing ---
if not exist "%PYTHON_EXE%" (
    echo Downloading Python %PY_VERSION% embeddable...
    powershell -Command "Invoke-WebRequest '%PY_URL%' -OutFile '%PY_ZIP%'"
    if errorlevel 1 (
        echo ERROR: Python download failed
        pause
        exit /b 1
    )
    echo Extracting Python...
    powershell -Command "Expand-Archive '%PY_ZIP%' '%PYTHON_DIR%' -Force"
    del "%PY_ZIP%"
)



REM --- Download get-pip.py if missing ---
if not exist "%GET_PIP_FILE%" (
    echo Downloading get-pip.py...
    powershell -Command "Invoke-WebRequest '%GET_PIP_URL%' -OutFile '%GET_PIP_FILE%'"
    if errorlevel 1 (
        echo ERROR: get-pip.py download failed
        pause
        exit /b 1
    )
)

REM --- Configure python._pth ---
set PTH_FILE=%PYTHON_DIR%\python%PY_MAJOR%._pth
echo Configuring %PTH_FILE%
(
echo python%PY_MAJOR%.zip
echo Lib
echo Lib\site-packages
echo ..
echo import site
) > "%PTH_FILE%"

REM --- Install pip ---
"%PYTHON_EXE%" "%GET_PIP_FILE%"

REM --- Upgrade pip ---
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel

REM --- Install dependencies ---
"%PYTHON_EXE%" -m pip install -r "%ROOT_DIR%\requirements.txt" ^
 --extra-index-url https://download.pytorch.org/whl/cu128

REM --- Install metashape ---
"%PYTHON_EXE%" -m pip install "%METASHAPE_PATH%"

if errorlevel 1 (
    echo ERROR: dependency installation failed
    pause
    exit /b 1
)

REM --- Create RT.bat ---
set RUN_BAT=%ROOT_DIR%\RT.bat
echo Creating %RUN_BAT%...
(
echo @echo off
echo set ROOT_DIR=%LOCALAPPDATA%\Reprojection_toolbox
echo cd /d "%%ROOT_DIR%%"
echo "%%ROOT_DIR%%\python\python.exe" -m app
) > "%RUN_BAT%"

echo.
echo === Reprojection toolbox installation complete ===
echo You can now launch the app using RT.bat in %ROOT_DIR%
@echo off
set ROOT_DIR=%LOCALAPPDATA%\Reprojection_toolbox
cd /d "%ROOT_DIR%"
"%ROOT_DIR%\python\python.exe" -m app

