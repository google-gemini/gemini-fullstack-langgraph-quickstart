#!/bin/bash

# Kill ports to ensure clean start
fuser -k 8000/tcp > /dev/null 2>&1
fuser -k 5173/tcp > /dev/null 2>&1

echo "🚀 Starting Backend (FastAPI)..."
cd backend
# Run using uvicorn directly as seen in server.py structure, assuming src module path is correct
# We need to set PYTHONPATH to include src
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
nohup python3 src/server.py > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

echo "🚀 Starting Frontend (Vite)..."
cd frontend
# Ensure dependencies
if [ ! -d "node_modules" ]; then
    npm install > /dev/null 2>&1
fi
nohup npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "⏳ Waiting 15 seconds for services..."
sleep 15

echo "🔍 Verifying Backend Health..."
if curl -s http://127.0.0.1:8000/docs > /dev/null; then
    echo "✅ Backend is ONLINE!"
else
    echo "❌ Backend failed. Checking logs..."
    tail -n 20 backend.log
    kill $BACKEND_PID $FRONTEND_PID
    exit 1
fi

echo "🧠 Testing AI Logic (Accounting Query)..."
# Test query: Ask for summary of expenses
RESPONSE=$(curl -s -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "สรุปรายจ่ายทั้งหมดในบัญชีให้หน่อย"}')

echo "Response from AI:"
echo "$RESPONSE"

echo "---------------------------------------------------"
echo "✅ Setup Complete. Logs are in backend.log and frontend.log"
echo "You can access the UI at http://localhost:5173"