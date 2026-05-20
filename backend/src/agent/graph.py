import os
import json
import operator
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig

from agent.state import OverallState
from agent.configuration import Configuration
from agent.specialized_prompts import (
    SUPERVISOR_INSTRUCTIONS,
    CODING_AGENT_INSTRUCTIONS,
    VISION_AGENT_INSTRUCTIONS,
    DATA_AGENT_INSTRUCTIONS
)

load_dotenv()

def get_model(config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    return ChatGoogleGenerativeAI(
        model=configurable.query_generator_model, 
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2
    )

import re

# --- 1. Planner Node ---
def planner_node(state: OverallState, config: RunnableConfig):
    llm = get_model(config)
    user_request = state["messages"][-1].content
    
    prompt = f"""คุณคือ Senior AI Planner. จงย่อยคำสั่งของผู้ใช้เป็นขั้นตอนสั้นๆ (Micro-tasks) 
    เพื่อให้ทีม AI Specialist (Coder, Vision, Data, Researcher) ทำงานต่อกันเป็นสายพาน.
    
    คำสั่งผู้ใช้: {user_request}
    
    กฎการย่อยงาน:
    1. แต่ละขั้นตอนต้องจบในตัวและชัดเจน
    2. ระบุชื่อ Specialist ที่ควรทำในแต่ละขั้นตอนด้วยรูปแบบ [AgentName]: Task description
    3. ส่งกลับมาเป็น JSON array ของสตริงเท่านั้น ห้ามมีข้อความอื่นปน
    ตัวอย่าง: [" [Data]: อ่านข้อมูลจากไฟล์ CSV", "[Coder]: เขียนสคริปต์คำนวณกำไร"]
    """
    
    response = llm.invoke(prompt)
    try:
        # ใช้ Regex ค้นหา JSON array ([...]) เพื่อความแม่นยำ
        json_match = re.search(r"\[.*\]", response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            tasks = json.loads(json_str)
        else:
            # Fallback ถ้าหา JSON ไม่เจอจริงๆ
            tasks = [f"[Researcher]: {user_request}"]
    except Exception as e:
        print(f"Planner Parsing Error: {e}")
        tasks = [f"[Researcher]: {user_request}"]
        
    return {"task_queue": tasks, "current_step_index": 0, "active_agent": "Planner"}

# --- 2. Executor Node ---
def executor_node(state: OverallState, config: RunnableConfig):
    llm = get_model(config)
    task_with_agent = state["task_queue"][state["current_step_index"]]
    
    # แยกชื่อ Agent และตัวงานออก
    if "]:" in task_with_agent:
        agent_name, current_task = task_with_agent.split("]:", 1)
        agent_name = agent_name.replace("[", "").strip()
    else:
        agent_name = "Researcher"
        current_task = task_with_agent

    # เลือก System Prompt ตามประเภท Agent
    system_prompt = ""
    if "Coder" in agent_name:
        system_prompt = CODING_AGENT_INSTRUCTIONS
    elif "Vision" in agent_name:
        system_prompt = VISION_AGENT_INSTRUCTIONS
    elif "Data" in agent_name:
        system_prompt = DATA_AGENT_INSTRUCTIONS
    else:
        system_prompt = "You are a helpful Research Assistant."

    # Load accounting data if available
    accounting_context = ""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Try multiple possible locations for accounting_data.csv
    possible_paths = [
        os.path.abspath(os.path.join(current_dir, "../../../accounting_data.csv")), # Local
        "/deps/backend/accounting_data.csv", # Docker (if copied there)
        "/app/accounting_data.csv" # Generic Docker root
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = f.read()
                    accounting_context = f"\n\n--- ข้อมูลบัญชีปัจจุบัน (Accounting Data) ---\n{data}\n------------------------------------------\n"
                break
            except Exception:
                continue

    full_prompt = f"{system_prompt}\n\nContext:\n{accounting_context}\nPrevious Progress: {state.get('last_output', 'None')}\n\nCurrent Task to execute: {current_task}"
    result = llm.invoke(full_prompt)
    
    return {
        "last_output": result.content, 
        "messages": [AIMessage(content=f"[{agent_name}]: {result.content}")],
        "active_agent": agent_name
    }

# --- 3. Verifier Node ---
def verifier_node(state: OverallState, config: RunnableConfig):
    llm = get_model(config)
    work_to_check = state["last_output"]
    task_description = state["task_queue"][state["current_step_index"]]
    
    prompt = f"""คุณคือ Quality Assurance. ตรวจสอบงานนี้อย่างเข้มงวด:
    งานที่ได้รับมอบหมาย: {task_description}
    ผลลัพธ์ที่ AI ทำออกมา: {work_to_check}
    
    เกณฑ์การตัดสิน:
    - ถ้างานถูกต้อง ครบถ้วนตามสั่ง 100% ให้ตอบเพียงคำเดียวว่า 'PASSED'
    - ถ้ามีจุดผิดพลาด หรือไม่ครบถ้วน ให้ตอบ 'FAILED' ตามด้วยเหตุผลและสิ่งที่ต้องแก้ไข
    """
    
    response = llm.invoke(prompt)
    is_passed = "PASSED" in response.content.upper()
    
    return {
        "verification_passed": is_passed, 
        "error_feedback": response.content if not is_passed else "",
        "active_agent": "Verifier"
    }

# --- 4. Assembly Line Router ---
def assembly_line_router(state: OverallState):
    if not state["verification_passed"]:
        return "executor" # ตีกลับไปแก้
    
    if state["current_step_index"] + 1 < len(state["task_queue"]):
        state["current_step_index"] += 1
        return "executor" # ไปงานถัดไป
    
    return END

# --- Build the Graph ---
builder = StateGraph(OverallState, config_schema=Configuration)

builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("verifier", verifier_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "verifier")

builder.add_conditional_edges("verifier", assembly_line_router, {
    "executor": "executor",
    END: END
})

graph = builder.compile()

