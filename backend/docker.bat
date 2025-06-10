@echo off
title Docker Services Launcher
color 0A

echo [BAT1] Đang khởi chạy backend và frontend...
start "" "backend.bat"
start "" "frontend.bat"

echo [BAT1] Đang khởi động Docker services...
cd /d D:\Python\smart_traffic_iot\backend\.docker
docker compose up



pause