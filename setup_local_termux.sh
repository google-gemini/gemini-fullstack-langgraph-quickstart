#!/bin/bash

echo "🚀 Starting Local Setup for Gemini Fullstack Agent (Termux Edition)..."

# 1. Backend Setup
echo "---------------------------------------------------"
echo "🐍 Setting up Backend (Python)..."
cd backend
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Created backend/.env. Please edit it and add your GEMINI_API_KEY!"
fi

# Install Dependencies (Lightweight mode)
echo "📦 Installing Python dependencies..."
pip install -q -e .
pip install -q "langgraph-cli[inmem]"
cd ..

# 2. Frontend Setup
echo "---------------------------------------------------"
echo "⚛️  Setting up Frontend (React)..."
cd frontend
if [ ! -d node_modules ]; then
    echo "📦 Installing Node modules (this might take a moment)..."
    npm install
fi
cd ..

# 3. Final Instructions
echo "---------------------------------------------------"
echo "✅ Setup Complete!"
echo ""
echo "🔥 To run the system, you need TWO Termux sessions:"
echo ""
echo "👉 Session 1 (Backend):"
echo "   cd gemini-fullstack-langgraph-quickstart/backend"
echo "   export GEMINI_API_KEY=your_actual_api_key_here  (if not in .env)"
echo "   langgraph dev"
echo ""
echo "👉 Session 2 (Frontend):"
echo "   cd gemini-fullstack-langgraph-quickstart/frontend"
echo "   npm run dev"
echo ""
echo "🌍 Open http://localhost:5173 in your browser to start chatting!"
