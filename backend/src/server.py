import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from agent.graph import graph
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Universal AI Agent API")

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # เตรียม Initial State
        inputs = {"messages": [HumanMessage(content=request.message)]}
        
        # รัน Graph จนจบ
        result = graph.invoke(inputs)
        
        # รวบรวมคำตอบจากทุกลำดับในสายพาน
        responses = []
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                responses.append(msg.content)
        
        return {
            "status": "success",
            "final_answer": responses[-1] if responses else "No response",
            "full_chain": responses,
            "tasks_completed": result.get("task_queue", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
