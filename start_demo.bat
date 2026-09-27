@echo off
rem DepthWizard demo: double-click to start (see start_demo.ps1). Add -Offline for no internet.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_demo.ps1" %*
