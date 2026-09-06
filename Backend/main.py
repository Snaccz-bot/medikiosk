"""
MediKiosk — Production Outpatient Triage Backend
Ministry of AYUSH / AIIA Hackathon
"""

import os
import json
import uuid
import re
import io
import base64
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import requests
from PIL import Image

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

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip().strip("\"'")
if GEMINI_KEY:
    print(f"\n[MediKiosk] ✓ Google Gemini API Key configured ({GEMINI_KEY[:8]}...)")
else:
    print("\n[MediKiosk] ⚠ Running with clinical heuristic fallback")

def clean_json_str(raw: str) -> str:
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
    clean = re.sub(r"\s*```$", "", clean).strip()
    first = clean.find("{")
    last = clean.rfind("}")
    if first != -1 and last != -1:
        clean = clean[first:last+1]
    return clean

# Single-credential REST call (resolves 400 Multiple Credentials error)
def query_gemini_text(prompt: str) -> dict:
    if not GEMINI_KEY:
        raise RuntimeError("No Gemini API key configured")
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_KEY
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=8)
        if r.status_code == 200:
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(clean_json_str(text))
        else:
            print(f"[Gemini Text Error {r.status_code}]: {r.text[:150]}")
    except Exception as e:
        print(f"[Gemini Text Exception]: {e}")
    raise RuntimeError("Gemini text query failed")

def query_gemini_vision(file_bytes: bytes, content_type: str, prompt: str) -> dict:
    if not GEMINI_KEY:
        raise RuntimeError("No Gemini API key configured")

    # Fast downscale to prevent upload timeouts
    if content_type.startswith("image/"):
        try:
            img = Image.open(io.BytesIO(file_bytes))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail((800, 800))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70, optimize=True)
            file_bytes = buf.getvalue()
            content_type = "image/jpeg"
        except Exception:
            pass

    b64_data = base64.b64encode(file_bytes).decode("utf-8")
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_KEY
    }
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": content_type, "data": b64_data}},
                {"text": prompt}
            ]
        }]
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        if r.status_code == 200:
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(clean_json_str(text))
        else:
            print(f"[Gemini Vision Error {r.status_code}]: {r.text[:150]}")
    except Exception as e:
        print(f"[Gemini Vision Exception]: {e}")
    raise RuntimeError("Gemini vision query failed")


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
    except Exception:
        mongo_col = None

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
        except Exception:
            pass
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
    phone: str
    language: str = "en"
    mock_abha: Optional[str] = ""

class ChatRequest(BaseModel):
    session_id: str
    user_message: str

class SummaryRequest(BaseModel):
    session_id: str

class UpdateSummaryRequest(BaseModel):
    clinical_summary: dict


# ── 5. TOLERANT DOCTOR AUTHENTICATION ─────────────────────────────────────────
@app.post("/doctor/login")
def doctor_login(req: DoctorLoginRequest):
    doc_id = req.doctor_id.strip().upper()
    pin = req.pin.strip().lower()
    
    if doc_id in ("DOC1", "DOC-101", "DR.SHARMA") and pin in ("present", "1234"):
        return {
            "success": True,
            "doctor_name": "Dr. Rameshwar Sharma, MD (Ayu)",
            "role": "Senior Consultant",
            "department": "Kayachikitsa (Internal Medicine) OPD Room 2"
        }
    raise HTTPException(status_code=401, detail="Invalid Doctor ID or Password")


# ── 6. PATIENT KIOSK & ADAPTIVE INTAKE ────────────────────────────────────────
LANG_NAMES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil",
    "ml": "Malayalam", "gu": "Gujarati", "bn": "Bengali"
}

GREETINGS = {
    "hi", "hello", "hey", "hii", "k", "ok", "namaste", "vanakkam", 
    "namaskaram", "yo", "good morning", "good evening", "fine", "cool", "h", "no", "yes"
}

def clean_symptom_text(text: str) -> str:
    clean = text.strip()
    if clean.lower().strip("!., ") in GREETINGS:
        return ""
    return clean

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
    
    active_sessions = [
        s for s in sessions.values()
        if s.get("status") in ("ready", "in_consultation")
    ]
    active_sessions.sort(key=lambda x: x.get("created_at", ""))
    
    in_room = next((s for s in active_sessions if s.get("status") == "in_consultation"), None)
    current_token = in_room["token"] if in_room else "None in consultation"
    
    my_created = session.get("created_at", "")
    patients_ahead = sum(
        1 for s in active_sessions 
        if s.get("status") == "ready" and s.get("created_at", "") < my_created
    )
    
    est_wait = max(patients_ahead * 6, 2) if session.get("status") != "in_consultation" else 0
    
    return {
        "status": session.get("status", "intake"),
        "token": session.get("token"),
        "room": "OPD Consultation Room 2",
        "current_token_in_room": current_token,
        "patients_ahead": patients_ahead,
        "est_wait_minutes": est_wait
    }


@app.post("/chat/next")
def chat_next(req: ChatRequest):
    sessions = db_get_all()
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_text = req.user_message.strip()
    if user_text:
        session["messages"].append({"role": "user", "text": user_text})

    user_msgs = [m["text"] for m in session["messages"] if m["role"] == "user"]
    clinical_msgs = [t for t in user_msgs if clean_symptom_text(t)]
    clinical_count = len(clinical_msgs)

    lang_code = session["patient"].get("language", "en")
    target_lang = LANG_NAMES.get(lang_code, "English")

    # If the user only said "hi" or empty input, prompt for symptoms
    if clinical_count == 0:
        q = "Hello! Please describe your symptoms or what health problem brought you in today." if lang_code == "en" else "नमस्ते! कृपया बताइए आज आपको क्या तकलीफ़ या स्वास्थ्य समस्या है?"
        session["messages"].append({"role": "bot", "text": q})
        db_save(session)
        return {"question": q, "done": False}

    transcript = "\n".join([
        f"{'Assistant' if m['role']=='bot' else 'Patient'}: {m['text']}"
        for m in session["messages"]
    ])

    prompt = f"""You are MediKiosk, a pre-consultation OPD triage assistant at an Indian hospital.
The patient is describing their condition. Ask ONE short, relevant medical question (under 16 words) in {target_lang}.

Clinical Rules:
- If the patient just entered random letters or unintelligible input, ask them to describe where they feel pain or discomfort.
- If duration/onset is missing: Inquire how long they have had this symptom.
- If radiation/swelling is missing: Ask if it spreads or is accompanied by swelling or stiffness.
- Inquire about current medications and drug allergies.
- NEVER repeat a question.
- After 3 to 4 clinical responses, return done: true and an empty question.

JSON format:
{{"question": "<question string or empty if done>", "done": <true or false>}}

Conversation:
{transcript}"""

    try:
        parsed = query_gemini_text(prompt)
        question = parsed.get("question", "")
        done = bool(parsed.get("done", False)) or not question or clinical_count >= 4
    except Exception:
        # Dynamic fallback based on genuine clinical turns
        if clinical_count == 1:
            question = "How long have you had this, and does it spread or feel worse with movement?"
            done = False
        elif clinical_count == 2:
            question = "Do you have any associated swelling, morning stiffness, fever, or numbness?"
            done = False
        elif clinical_count == 3:
            question = "Are you currently taking any medicines or painkillers, and do you have any allergies?"
            done = False
        else:
            question = ""
            done = True

    if question:
        session["messages"].append({"role": "bot", "text": question})

    db_save(session)
    return {"question": question, "done": done}


# Standard threadpooled function (Prevents server lockup!)
@app.post("/upload/document")
def upload_document(session_id: str, file: UploadFile = File(...)):
    sessions = db_get_all()
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    file_bytes = file.file.read()
    content_type = file.content_type or "image/jpeg"
    if file.filename.lower().endswith(".pdf"):
        content_type = "application/pdf"
    elif file.filename.lower().endswith(".png"):
        content_type = "image/png"

    prompt = """Examine this medical prescription or report image.
Extract findings into JSON:
{
  "summary": "1 to 2 sentences summarizing the visible prescription medicines, dosages, or lab findings. If this is a medicine packaging or bottle, identify the medicine name and strength. If text is completely unreadable, write: 'Uploaded image inspected: Document attached for doctor's in-person review.'",
  "medicines": ["List of medicines found with dosage"],
  "tests": ["Key lab findings if present"],
  "date": "Document date if found"
}"""

    try:
        extracted = query_gemini_vision(file_bytes, content_type, prompt)
    except Exception as e:
        print(f"[Vision Fallback]: {e}")
        extracted = {
            "summary": f"Medical document ({file.filename}) uploaded and attached for doctor's review.",
            "medicines": [],
            "tests": [],
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
    doc_summary_text = upload_data.get("summary", "No document uploaded.")
    doc_data = json.dumps(upload_data)
    
    lang_code = session["patient"].get("language", "en")
    target_lang = LANG_NAMES.get(lang_code, "English")

    prompt = f"""You are MediKiosk's clinical clerk.
Synthesize the patient intake into a physician-ready case note.

Rules:
- Filter out greetings ('hi', 'hello', 'ok') and random non-words. The 'chiefComplaint' must reflect the patient's actual reported concern.
- Write clinical fields in concise medical English.
- If patient answered 'no' or 'none' to medicines/allergies, write 'None reported'.
- Under 'patientRecap', write exactly 3 short friendly bullets in {target_lang} for the patient.

JSON Schema:
{{
  "chiefComplaint": "...",
  "hpi": "...",
  "allergies": "...",
  "medications": "...",
  "uploadedDocSummary": "...",
  "clinicalImpression": "...",
  "patientRecap": ["bullet 1", "bullet 2", "bullet 3"]
}}

Transcript:
{transcript}

Document Findings:
{doc_data}"""

    try:
        parsed = query_gemini_text(prompt)
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            parsed = parsed[0]
    except Exception:
        clinical_msgs = [
            m["text"] for m in session["messages"] 
            if m["role"] == "user" and clean_symptom_text(m["text"])
        ]
        cc = clinical_msgs[0] if len(clinical_msgs) > 0 else "Patient OPD Consultation Request"
        timeline_notes = ". ".join(clinical_msgs[1:3]) if len(clinical_msgs) > 1 else "Reported during intake."
        
        last_str = " ".join(clinical_msgs[3:]).lower() if len(clinical_msgs) > 3 else ""
        if "no" in last_str or "none" in last_str:
            meds = "None reported"
            allergies = "None reported"
        else:
            meds = clinical_msgs[-1] if len(clinical_msgs) > 3 else "None reported"
            allergies = "Penicillin (Flagged)" if "penicillin" in transcript.lower() else "None reported"

        parsed = {
            "chiefComplaint": cc,
            "hpi": f"Patient presents with {cc}. Timeline & Details: {timeline_notes}",
            "allergies": allergies,
            "medications": meds,
            "uploadedDocSummary": doc_summary_text,
            "clinicalImpression": f"Clinical triage assessment for {cc}. Correlate with physical examination.",
            "patientRecap": [
                f"Complaint noted: {cc[:50]}",
                "Symptom timeline prepared for your doctor.",
                "Your case file is ready for OPD consultation."
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
                "red_flags": (c or {}).get("allergies", "")
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
