# In-memory database (works for demo, no setup needed)
db = {
    "judgments": {},
    "tasks": []
}

# Type hints for reference
class Task:
    id: str
    judgment_id: str
    action: str
    department: str
    deadline_phrase: str
    due_date: str
    internal_deadline: str
    priority: str  # high | medium | low
    status: str    # pending_approval | approved | in_progress | completed
    at_risk: bool
    created_at: str

class Judgment:
    id: str
    case_title: str
    case_number: str
    court: str
    judgment_date: str
    summary: str
    uploaded_at: str
    status: str    # pending_review | active | archived