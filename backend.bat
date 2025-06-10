@echo off
title Backend Server
color 0B

echo [BAT2] Đang khởi động backend server...
cd backend
uvicorn backend_launch:app --reload
pause