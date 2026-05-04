from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pdfplumber
import json
import os
import io
from groq import Groq
from dotenv import load_dotenv
from models import Task, Judgment, db
from datetime import datetime, timedelta
import uuid

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def extract_text_from_pdf(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def parse_judgment_with_ai(text: str) -> dict:
    prompt = f"""You are an expert at reading Indian court judgments and government orders.

Extract ALL directives, orders, and instructions from this judgment.
For each directive, extract:
1. The specific action required
2. The responsible department or officer (guess if not stated explicitly)
3. The deadline (convert vague phrases like "within reasonable time" = 30 days, "forthwith" = 7 days, "at the earliest" = 14 days, "immediately" = 3 days)
4. Priority: high (urgent legal compliance), medium (standard order), low (administrative)

Return ONLY valid JSON in this exact format, nothing else:
{{
  "case_title": "case name or 'Unknown Case'",
  "case_number": "case number or 'Unknown'",
  "court": "court name",
  "judgment_date": "date in YYYY-MM-DD or today's date",
  "summary": "2-3 sentence summary of the judgment",
  "directives": [
    {{
      "id": "unique_id_1",
      "action": "Clear description of what must be done",
      "department": "Responsible department/officer",
      "deadline_days": 30,
      "deadline_phrase": "original phrase from judgment",
      "priority": "high|medium|low"
    }}
  ]
}}

JUDGMENT TEXT:
{text[:4000]}"""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
    )

    response_text = chat_completion.choices[0].message.content.strip()

    # Clean up response if it has markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    return json.loads(response_text)


@app.post("/api/upload")
async def upload_judgment(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    file_bytes = await file.read()

    # Extract text
    text = extract_text_from_pdf(file_bytes)
    if not text or len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF. Try a text-based PDF.")

    # Parse with AI
    parsed = parse_judgment_with_ai(text)

    # Create judgment record
    judgment_id = str(uuid.uuid4())
    judgment_date = parsed.get("judgment_date", datetime.now().strftime("%Y-%m-%d"))

    judgment = {
        "id": judgment_id,
        "case_title": parsed.get("case_title", "Unknown Case"),
        "case_number": parsed.get("case_number", "Unknown"),
        "court": parsed.get("court", "Unknown Court"),
        "judgment_date": judgment_date,
        "summary": parsed.get("summary", ""),
        "uploaded_at": datetime.now().isoformat(),
        "status": "pending_review",
        "raw_text": text[:2000]
    }

    db["judgments"][judgment_id] = judgment

    # Create tasks
    tasks = []
    for directive in parsed.get("directives", []):
        deadline_days = directive.get("deadline_days", 30)
        due_date = datetime.now() + timedelta(days=deadline_days)
        internal_deadline = due_date - timedelta(days=7)

        task = {
            "id": str(uuid.uuid4()),
            "judgment_id": judgment_id,
            "action": directive.get("action", ""),
            "department": directive.get("department", "Concerned Department"),
            "deadline_phrase": directive.get("deadline_phrase", ""),
            "due_date": due_date.strftime("%Y-%m-%d"),
            "internal_deadline": internal_deadline.strftime("%Y-%m-%d"),
            "priority": directive.get("priority", "medium"),
            "status": "pending_approval",
            "created_at": datetime.now().isoformat(),
        }
        db["tasks"].append(task)
        tasks.append(task)

    return {
        "judgment": judgment,
        "tasks": tasks,
        "message": f"Found {len(tasks)} directives. Please review and approve."
    }


@app.get("/api/judgments")
async def get_judgments():
    return list(db["judgments"].values())


@app.get("/api/tasks")
async def get_tasks():
    tasks = db["tasks"]
    # Sort: high priority first, then by due date
    priority_order = {"high": 0, "medium": 1, "low": 2}
    tasks_sorted = sorted(tasks, key=lambda t: (
        priority_order.get(t["priority"], 1),
        t["due_date"]
    ))

    # Add risk flag
    today = datetime.now()
    for task in tasks_sorted:
        if task["status"] not in ("completed", "pending_approval"):
            due = datetime.strptime(task["due_date"], "%Y-%m-%d")
            created = datetime.fromisoformat(task["created_at"])
            total_days = (due - created).days or 1
            elapsed_days = (today - created).days
            pct = elapsed_days / total_days
            task["at_risk"] = pct > 0.7 and task["status"] == "in_progress"
        else:
            task["at_risk"] = False

    return tasks_sorted


@app.patch("/api/tasks/{task_id}/approve")
async def approve_task(task_id: str):
    for task in db["tasks"]:
        if task["id"] == task_id:
            task["status"] = "approved"
            return task
    raise HTTPException(status_code=404, detail="Task not found")


@app.patch("/api/tasks/{task_id}/status")
async def update_task_status(task_id: str, body: dict):
    for task in db["tasks"]:
        if task["id"] == task_id:
            task["status"] = body.get("status", task["status"])
            return task
    raise HTTPException(status_code=404, detail="Task not found")


@app.get("/api/alerts")
async def get_alerts():
    alerts = []
    today = datetime.now()
    for task in db["tasks"]:
        if task["status"] in ("completed",):
            continue
        internal = datetime.strptime(task["internal_deadline"], "%Y-%m-%d")
        due = datetime.strptime(task["due_date"], "%Y-%m-%d")
        days_to_internal = (internal - today).days
        days_to_due = (due - today).days

        if days_to_due < 0:
            alerts.append({"task_id": task["id"], "action": task["action"], "type": "overdue", "message": f"OVERDUE by {abs(days_to_due)} days"})
        elif days_to_internal <= 0:
            alerts.append({"task_id": task["id"], "action": task["action"], "type": "urgent", "message": f"Court deadline in {days_to_due} days"})
        elif days_to_internal <= 3:
            alerts.append({"task_id": task["id"], "action": task["action"], "type": "warning", "message": f"Internal deadline in {days_to_internal} days"})

    return alerts


@app.get("/api/stats")
async def get_stats():
    tasks = db["tasks"]
    total = len(tasks)
    approved = sum(1 for t in tasks if t["status"] == "approved")
    pending = sum(1 for t in tasks if t["status"] == "pending_approval")
    completed = sum(1 for t in tasks if t["status"] == "completed")
    high_priority = sum(1 for t in tasks if t["priority"] == "high" and t["status"] != "completed")

    return {
        "total": total,
        "approved": approved,
        "pending_approval": pending,
        "completed": completed,
        "high_priority": high_priority,
        "judgments": len(db["judgments"])
    }