"""
MediKiosk — Unified Full-Stack Backend
Serves both API and Frontend at http://localhost:8000
"""

import os
import json
import uuid
import re
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="MediKiosk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 1. ENVIRONMENT & KEYS ─────────────────────────────────────────────────────
env_file = Path(".env")
if not env_file.exists():
    env_file = Path(__file__).parent / ".env"

if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip("\"'")

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_KEY:
    print(f"\n[MediKiosk] ✓ Google Gemini API Key loaded ({GEMINI_KEY[:8]}...)")
    genai.configure(api_key=GEMINI_KEY)
else:
    print("\n[MediKiosk] ⚠ Running with clinical heuristic agent (No Gemini Key)")

MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

def query_gemini(prompt_or_contents):
    last_err = None
    for model_name in MODELS:
        try:
            m = genai.GenerativeModel(model_name)
            res = m.generate_content(
                prompt_or_contents,
                generation_config={"response_mime_type": "application/json"}
            )
            raw = res.text.strip()
            clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
            clean = re.sub(r"\s*```$", "", clean).strip()
            first = clean.find("{")
            last = clean.rfind("}")
            if first != -1 and last != -1:
                clean = clean[first:last+1]
            return json.loads(clean)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Gemini call failed: {last_err}")


# ── 2. DATABASE LAYER ─────────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "")
mongo_col = None

if MONGO_URI:
    try:
        from pymongo import MongoClient
        mclient = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
        mclient.admin.command('ping')
        mongo_col = mclient["medikiosk"]["sessions"]
        print("[MediKiosk] ✓ Connected to MongoDB Atlas Cloud Database!\n")
    except Exception as e:
        print(f"[MediKiosk] ⚠ MongoDB connection failed ({e}). Using local sessions.json\n")
        mongo_col = None
else:
    print("[MediKiosk] ℹ MONGO_URI not set. Using local sessions.json\n")

DB_FILE = Path("sessions.json")

def db_get_all() -> dict:
    if mongo_col is not None:
        try:
            docs = list(mongo_col.find({}, {"_id": 0}))
            return {d["session_id"]: d for d in docs if "session_id" in d}
        except Exception:
            pass
    if not DB_FILE.exists():
        return {}
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def db_save(session: dict):
    if mongo_col is not None:
        try:
            mongo_col.replace_one({"session_id": session["session_id"]}, session, upsert=True)
        except Exception as e:
            print(f"Mongo write error: {e}")
    all_data = db_get_all()
    all_data[session["session_id"]] = session
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

def db_delete(session_id: str):
    if mongo_col is not None:
        try:
            mongo_col.delete_one({"session_id": session_id})
        except Exception:
            pass
    all_data = db_get_all()
    if session_id in all_data:
        del all_data[session_id]
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)


# ── 3. SERVE FRONTEND AT http://localhost:8000 ───────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_home():
    candidate_paths = [
        Path(__file__).parent.parent / "Frontend" / "index.html",
        Path(__file__).parent / "Frontend" / "index.html",
        Path(__file__).parent / "index.html",
        Path("Frontend/index.html"),
        Path("index.html")
    ]
    for p in candidate_paths:
        if p.exists():
            return FileResponse(p)
    return HTMLResponse("<h2>MediKiosk: index.html not found.</h2>")


# ── 4. DATA MODELS ────────────────────────────────────────────────────────────
class DoctorLoginRequest(BaseModel):
    doctor_id: str
    pin: str

class StartSessionRequest(BaseModel):
    name: str
    age: str
    gender: str
    phone: Optional[str] = ""
    language: str = "en"
    mock_abha: Optional[str] = ""

class ChatRequest(BaseModel):
    session_id: str
    user_message: str

class SummaryRequest(BaseModel):
    session_id: str

class UpdateSummaryRequest(BaseModel):
    clinical_summary: dict


# ── 5. DOCTOR AUTHENTICATION ──────────────────────────────────────────────────
VALID_DOCTORS = {
    "DOC1": "Present",
    "DOC-101": "1234",
    "DR.SHARMA": "1234"
}

@app.post("/doctor/login")
def doctor_login(req: DoctorLoginRequest):
    doc_id = req.doctor_id.strip().upper()
    pin = req.pin.strip()
    
    # Check DOC1 with Present (handles case-insensitivity as safety)
    if doc_id == "DOC1" and pin.lower() == "present":
        return {
            "success": True,
            "doctor_name": "Dr. Rameshwar Sharma, MD (Ayu)",
            "role": "Senior Consultant",
            "department": "Kayachikitsa (Internal Medicine) OPD Room 2"
        }
    if doc_id in VALID_DOCTORS and VALID_DOCTORS[doc_id] == pin:
        return {
            "success": True,
            "doctor_name": "Dr. Rameshwar Sharma, MD (Ayu)",
            "role": "Senior Consultant",
            "department": "Kayachikitsa (Internal Medicine) OPD Room 2"
        }
    raise HTTPException(status_code=401, detail="Invalid Doctor ID or Password")


# ── 6. PATIENT KIOSK ROUTES ───────────────────────────────────────────────────

@app.post("/session/start")
def start_session(req: StartSessionRequest):
    session_id = str(uuid.uuid4())
    token = f"B-{10 + int(uuid.uuid4().int % 40)}"

    session = {
        "session_id": session_id,
        "token": token,
        "patient": req.model_dump(),
        "messages": [],
        "upload_summary": None,
        "clinical_summary": None,
        "status": "intake",
        "created_at": datetime.utcnow().isoformat()
    }
    db_save(session)
    return {"session_id": session_id, "token": token}


@app.get("/session/{session_id}/status")
def get_session_status(session_id: str):
    sessions = db_get_all()
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "status": session.get("status", "intake"),
        "token": session.get("token"),
        "room": "OPD Consultation Room 2"
    }


@app.post("/chat/next")
def chat_next(req: ChatRequest):
    sessions = db_get_all()
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if req.user_message.strip():
        session["messages"].append({"role": "user", "text": req.user_message.strip()})

    user_msgs = [m for m in session["messages"] if m["role"] == "user"]
    msg_count = len(user_msgs)

    transcript = "\n".join([
        f"{'Assistant' if m['role']=='bot' else 'Patient'}: {m['text']}"
        for m in session["messages"]
    ]) or "(Patient has arrived at kiosk.)"

    lang_inst = "in Hindi" if session["patient"].get("language") == "hi" else "in English"

    prompt = f"""You are MediKiosk, an outpatient triage assistant at an Indian hospital OPD.
Ask ONE short, empathetic medical question (under 18 words) {lang_inst} to build a doctor-ready case history.
Do NOT repeat topics the patient already mentioned.
After 3 to 4 patient answers, return done: true and an empty question.

JSON format:
{{"question": "<question string or empty if done>", "done": <true or false>}}

Conversation:
{transcript}"""

    try:
        parsed = query_gemini(prompt)
        question = parsed.get("question", "")
        done = bool(parsed.get("done", False)) or not question
    except Exception:
        if msg_count == 0:
            question = "नमस्ते, बताइए क्या तकलीफ़ है?" if session["patient"].get("language") == "hi" else "How can we help you today?"
            done = False
        elif msg_count == 1:
            question = "यह दर्द या तकलीफ़ कब से है, और किस समय ज़्यादा बढ़ती है?" if session["patient"].get("language") == "hi" else "Since when have you had this, and does anything make it worse?"
            done = False
        elif msg_count == 2:
            question = "क्या आपको कोई सूजन, बुखार या अन्य लक्षण महसूस होते हैं?" if session["patient"].get("language") == "hi" else "Do you have any swelling, morning stiffness, or other symptoms?"
            done = False
        elif msg_count == 3:
            question = "क्या आप वर्तमान में कोई दवा ले रहे हैं या किसी दवा से एलर्जी है?" if session["patient"].get("language") == "hi" else "Are you currently taking any medications, or do you have any allergies?"
            done = False
        else:
            question = ""
            done = True

    if question:
        session["messages"].append({"role": "bot", "text": question})

    db_save(session)
    return {"question": question, "done": done}


@app.post("/upload/document")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    sessions = db_get_all()
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    file_bytes = await file.read()
    content_type = file.content_type or "image/jpeg"
    if file.filename.lower().endswith(".pdf"):
        content_type = "application/pdf"
    elif file.filename.lower().endswith(".png"):
        content_type = "image/png"

    prompt = """Analyze this prescription or report. Extract into JSON:
{
  "summary": "1-sentence summary of findings or diagnosis",
  "medicines": ["List of medicine names, doses, and frequencies"],
  "tests": ["Key lab test names and values"],
  "date": "Document date if found, else 'Recent'"
}"""

    try:
        doc_part = {"mime_type": content_type, "data": file_bytes}
        extracted = query_gemini([doc_part, prompt])
    except Exception:
        extracted = {
            "summary": f"{file.filename} logged for consultation.",
            "medicines": ["Previous prescription noted"],
            "tests": ["Report recorded"],
            "date": "Recent"
        }

    session["upload_summary"] = extracted
    db_save(session)
    return {"extracted": extracted}


@app.post("/summary/generate")
def generate_summary(req: SummaryRequest):
    sessions = db_get_all()
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    transcript = "\n".join([f"{m['role']}: {m['text']}" for m in session["messages"]])
    upload_data = session.get("upload_summary") or {}
    doc_data = json.dumps(upload_data)
    lang = "Hindi" if session["patient"].get("language") == "hi" else "English"

    prompt = f"""You are a clinical summarizer for hospital OPDs.
Based on the transcript and document data, produce a structured case note in JSON.
Rules:
- Write clinical fields in concise medical English.
- Flag allergies or urgent issues in 'redFlags'.
- Write 'patientRecap' as 3 short friendly bullets in {lang}.

JSON format:
{{
  "chiefComplaint": "...",
  "hpi": "...",
  "pastHistory": "...",
  "allergies": "...",
  "medications": "...",
  "familyHistory": "...",
  "reviewOfSystems": "...",
  "redFlags": "...",
  "uploadedDocSummary": "...",
  "patientRecap": ["bullet 1", "bullet 2", "bullet 3"]
}}

Transcript:
{transcript}

Document:
{doc_data}"""

    try:
        parsed = query_gemini(prompt)
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            parsed = parsed[0]
    except Exception:
        user_msgs = [m["text"] for m in session["messages"] if m["role"] == "user"]
        cc = user_msgs[0] if len(user_msgs) > 0 else "Consultation requested"
        duration_note = user_msgs[1] if len(user_msgs) > 1 else "Onset reported by patient."
        symptoms_note = user_msgs[2] if len(user_msgs) > 2 else "Symptoms reported."
        meds_note = user_msgs[3] if len(user_msgs) > 3 else "None reported."

        all_text = " ".join(user_msgs).lower()
        allergy_flag = "Allergic to Penicillin — Alert doctor." if "penicillin" in all_text else "None reported."
        past_hx = "High Blood Pressure reported." if "bp" in all_text or "pressure" in all_text else "None reported."
        current_meds = "Paracetamol (occasional)" if "paracetamol" in all_text else meds_note

        parsed = {
            "chiefComplaint": cc,
            "hpi": f"{cc}. {duration_note}. Associated: {symptoms_note}",
            "pastHistory": past_hx,
            "allergies": allergy_flag,
            "medications": current_meds,
            "familyHistory": "Not contributory.",
            "reviewOfSystems": "Normal aside from chief complaint.",
            "redFlags": allergy_flag if allergy_flag != "None reported." else "None flagged.",
            "uploadedDocSummary": upload_data.get("summary", "No document uploaded."),
            "patientRecap": [
                f"Complaint noted: {cc[:50]}...",
                "Onset, duration, and stiffness details recorded for doctor.",
                "Vitals and medications logged for consultation."
            ]
        }

    patient_recap = parsed.pop("patientRecap", [
        "Complaint and symptoms recorded for consultation.",
        "Medical history prepared for OPD doctor.",
        "Your case file is now ready in the consultation room."
    ])

    session["clinical_summary"] = parsed
    session["patient_recap"] = patient_recap
    session["status"] = "ready"
    db_save(session)

    return {
        "clinical_summary": parsed,
        "patient_recap": patient_recap,
        "token": session["token"]
    }


# ── 7. DOCTOR OPD MANAGEMENT ROUTES ───────────────────────────────────────────

@app.get("/doctor/queue")
def doctor_queue():
    sessions = db_get_all()
    queue = []
    for s in sessions.values():
        c = s.get("clinical_summary")
        status = s.get("status", "intake")
        if status != "completed":
            queue.append({
                "session_id": s["session_id"],
                "token": s["token"],
                "name": s["patient"].get("name", "—"),
                "age": s["patient"].get("age", "—"),
                "gender": s["patient"].get("gender", "—"),
                "status": status,
                "meta": f"Token: {s['token']} · {'Ready' if c else 'In Intake'}",
                "clinical": c,
                "red_flags": (c or {}).get("redFlags", "")
            })
    queue.reverse()
    return {"queue": queue}


@app.patch("/doctor/session/{session_id}/summary")
def update_summary(session_id: str, req: UpdateSummaryRequest):
    sessions = db_get_all()
    if session_id in sessions:
        session = sessions[session_id]
        session["clinical_summary"] = req.clinical_summary
        db_save(session)
        return {"status": "saved"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/doctor/session/{session_id}/approve")
def approve_patient(session_id: str):
    sessions = db_get_all()
    if session_id in sessions:
        session = sessions[session_id]
        session["status"] = "in_consultation"
        db_save(session)
        return {"status": "approved", "token": session["token"]}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/doctor/session/{session_id}/complete")
def complete_patient(session_id: str):
    sessions = db_get_all()
    if session_id in sessions:
        session = sessions[session_id]
        session["status"] = "completed"
        db_save(session)
        return {"status": "completed"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/doctor/session/{session_id}")
def delete_patient(session_id: str):
    sessions = db_get_all()
    if session_id in sessions:
        db_delete(session_id)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")
