@echo off
title Backend Server
color 0B

echo [BAT2] Đang khởi động backend server...
cd /d D:\Python\smart_traffic_iot\backend
uvicorn backend_launch:app --reload
pause