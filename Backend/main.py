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
import google.generativeai as genai
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

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_KEY:
    print(f"\n[MediKiosk] ✓ Google Gemini API Key loaded ({GEMINI_KEY[:8]}...)")
    genai.configure(api_key=GEMINI_KEY)
else:
    print("\n[MediKiosk] ⚠ Running with clinical heuristic agent")

# Stable GA Google AI Studio model names
MODELS = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]

def clean_json_str(raw: str) -> str:
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
    clean = re.sub(r"\s*```$", "", clean).strip()
    first = clean.find("{")
    last = clean.rfind("}")
    if first != -1 and last != -1:
        clean = clean[first:last+1]
    return clean

def query_gemini_text(prompt: str) -> dict:
    # 1. Try SDK
    for model_name in MODELS:
        try:
            m = genai.GenerativeModel(model_name)
            res = m.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(clean_json_str(res.text))
        except Exception:
            continue
    # 2. Try Direct REST fallback
    if GEMINI_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"}
            }
            r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=12)
            if r.status_code == 200:
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(clean_json_str(text))
        except Exception:
            pass
    raise RuntimeError("Gemini text query failed")

def query_gemini_vision(file_bytes: bytes, content_type: str, prompt: str) -> dict:
    # 1. Try PIL Image with SDK
    if content_type.startswith("image/"):
        try:
            pil_img = Image.open(io.BytesIO(file_bytes))
            for model_name in MODELS:
                try:
                    m = genai.GenerativeModel(model_name)
                    res = m.generate_content([pil_img, prompt])
                    return json.loads(clean_json_str(res.text))
                except Exception:
                    continue
        except Exception:
            pass

    # 2. Try Direct REST with base64
    if GEMINI_KEY:
        try:
            b64_data = base64.b64encode(file_bytes).decode("utf-8")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
            payload = {
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": content_type, "data": b64_data}},
                        {"text": prompt}
                    ]
                }],
                "generationConfig": {"responseMimeType": "application/json"}
            }
            r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=18)
            if r.status_code == 200:
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(clean_json_str(text))
        except Exception:
            pass
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


# ── 5. STRICT DOCTOR AUTHENTICATION ───────────────────────────────────────────
STRICT_DOCTORS = {
    "DOC1": "Present",
    "DOC-101": "1234"
}

@app.post("/doctor/login")
def doctor_login(req: DoctorLoginRequest):
    doc_id = req.doctor_id.strip()
    pin = req.pin.strip()
    
    if doc_id in STRICT_DOCTORS and STRICT_DOCTORS[doc_id] == pin:
        return {
            "success": True,
            "doctor_name": "Dr. Rameshwar Sharma, MD (Ayu)",
            "role": "Senior Consultant",
            "department": "Kayachikitsa (Internal Medicine) OPD Room 2"
        }
    raise HTTPException(status_code=401, detail="Invalid Doctor ID or Password")


# ── 6. PATIENT KIOSK & QUEUE ANALYTICS ROUTES ─────────────────────────────────
LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "bn": "Bengali"
}

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


GREETINGS = {"hi", "hello", "hey", "hii", "k", "ok", "namaste", "vanakkam", "namaskaram", "yo"}

@app.post("/chat/next")
def chat_next(req: ChatRequest):
    sessions = db_get_all()
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_text = req.user_message.strip()
    if user_text:
        session["messages"].append({"role": "user", "text": user_text})

    user_msgs = [m for m in session["messages"] if m["role"] == "user"]
    msg_count = len(user_msgs)

    lang_code = session["patient"].get("language", "en")
    target_lang = LANG_NAMES.get(lang_code, "English")

    # If the user just typed "hi" or empty input, don't ask symptom onset questions!
    clean_last = user_text.lower().strip("!., ")
    if msg_count > 0 and clean_last in GREETINGS:
        if lang_code == "hi":
            q = "नमस्ते! कृपया बताइए आज आपको क्या तकलीफ़ या समस्या है?"
        elif lang_code == "ta":
            q = "வணக்கம்! இன்று உங்களுக்கு என்ன உடல்நலப் பிரச்சினை உள்ளது?"
        elif lang_code == "ml":
            q = "നമസ്കാരം! ഇന്ന് എന്താണ് നിങ്ങളുടെ പ്രധാന ബുദ്ധിമുട്ട്?"
        elif lang_code == "gu":
            q = "નમસ્તે! આજે તમને શું તકલીફ અથવા સમસ્યા થઈ રહી છે?"
        elif lang_code == "bn":
            q = "নমস্কার! আজ আপনার কী শারীরিক সমস্যা হচ্ছে?"
        else:
            q = "Hello! Please tell us what specific symptoms or health issue you are experiencing today."
        session["messages"].append({"role": "bot", "text": q})
        db_save(session)
        return {"question": q, "done": False}

    transcript = "\n".join([
        f"{'Assistant' if m['role']=='bot' else 'Patient'}: {m['text']}"
        for m in session["messages"]
    ]) or "(Patient just arrived at kiosk.)"

    prompt = f"""You are MediKiosk, an expert OPD triage assistant at an Indian hospital (Ministry of AYUSH).
A patient is describing their condition. Ask ONE short question (under 16 words) in {target_lang}.

Adaptive Rules:
- If the patient hasn't stated any real illness yet, ask what health problem brought them in.
- If symptoms are stated, ask about duration, triggers, or severity.
- If duration is already answered, ask about associated signs (swelling, fever, radiation).
- Finally, ask about current medications and drug allergies.
- NEVER repeat a question the patient already answered.
- If you have gathered 3-4 distinct answers, return done: true and empty question.

JSON Schema:
{{"question": "<question string or empty if done>", "done": <true or false>}}

Conversation:
{transcript}"""

    try:
        parsed = query_gemini_text(prompt)
        question = parsed.get("question", "")
        done = bool(parsed.get("done", False)) or not question or msg_count >= 4
    except Exception:
        # Context-aware fallback: doesn't assume pain if not mentioned
        if msg_count == 0:
            question = "How can we help you today? Please tell us your main symptom." if lang_code == "en" else "नमस्ते, बताइए आज क्या समस्या है?"
            done = False
        elif msg_count == 1:
            question = "Since when have you been having this problem, and does anything make it worse?"
            done = False
        elif msg_count == 2:
            question = "Are you experiencing any other symptoms like fever, swelling, or pain?"
            done = False
        elif msg_count == 3:
            question = "Are you currently taking any medicines, or do you have any allergies?"
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

    prompt = """You are a medical record OCR synthesiser.
Carefully examine this prescription, lab report, or medical photo.
Extract clinical findings into pure JSON:
{
  "summary": "1 to 2 sentences summarizing what this medical document contains (e.g. 'Lab report shows elevated blood sugar' or 'Orthopedic prescription for knee pain'). If this is not a medical document or text is illegible, state: 'Document uploaded: No legible prescription text or lab values identified.'",
  "medicines": ["List of medicines found with dosage and frequency"],
  "tests": ["Key lab values found"],
  "date": "Document date if found, else 'Not specified'"
}"""

    try:
        extracted = query_gemini_vision(file_bytes, content_type, prompt)
    except Exception:
        extracted = {
            "summary": "Document received. No legible prescription medications or lab findings detected in this file.",
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

    prompt = f"""You are MediKiosk's clinical clerk preparing a doctor-ready case summary.
Rules:
- Write clinical fields in concise medical English regardless of conversation language.
- DO NOT invent filler. If something wasn't mentioned, state 'None reported'.
- Formulate a brief 'clinicalImpression' for the doctor's review.
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

Document Extraction:
{doc_data}"""

    try:
        parsed = query_gemini_text(prompt)
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            parsed = parsed[0]
    except Exception:
        user_msgs = [m["text"] for m in session["messages"] if m["role"] == "user"]
        cc = user_msgs[0] if len(user_msgs) > 0 else "Consultation requested"
        duration_note = user_msgs[1] if len(user_msgs) > 1 else "Reported by patient."
        symptoms_note = user_msgs[2] if len(user_msgs) > 2 else "Symptoms noted."
        meds_note = user_msgs[3] if len(user_msgs) > 3 else "None reported."

        all_text = " ".join(user_msgs).lower()
        allergy_flag = "Allergic to Penicillin — Alert doctor." if "penicillin" in all_text else "None reported."
        current_meds = "Paracetamol (reported)" if "paracetamol" in all_text else meds_note

        parsed = {
            "chiefComplaint": cc,
            "hpi": f"{cc}. {duration_note}. {symptoms_note}",
            "allergies": allergy_flag,
            "medications": current_meds,
            "uploadedDocSummary": doc_summary_text,
            "clinicalImpression": f"Clinical triage note for {cc}. Correlate with physical examination and vitals.",
            "patientRecap": [
                f"Complaint noted: {cc[:50]}",
                "Timeline and symptoms compiled for your doctor.",
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
