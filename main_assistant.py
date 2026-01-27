import os
import uuid
import glob
import threading
import time
import sqlite3
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Cookie, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import uvicorn
import requests
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs

# Import config and agents
from config import config
from agents.agent_decision import process_query
# Assuming MedicalRAG will be updated for learning later
from agents.rag_agent import MedicalRAG

# Initialize FastAPI app
app = FastAPI(title="FarmerASS - Agricultural Assistant", version="3.0")

# --- DATABASE SETUP (SQLite for Chat History) ---
DB_NAME = "chat_history.db"

def init_db():
    """Initialize SQLite database for chat sessions."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Table for Sessions (Chat threads)
    c.execute('''CREATE TABLE IF NOT EXISTS sessions 
                 (id TEXT PRIMARY KEY, title TEXT, created_at TEXT)''')
    # Table for Messages
    c.execute('''CREATE TABLE IF NOT EXISTS messages 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# --- HELPER FUNCTIONS ---

def save_message(session_id, role, content):
    """Save a single message to SQLite."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        timestamp = datetime.now().isoformat()
        c.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                  (session_id, role, content, timestamp))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving message: {e}")

def create_new_session(title="New Chat"):
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)", 
              (session_id, title, timestamp))
    conn.commit()
    conn.close()
    return session_id

def get_user_sessions():
    """Get all chat sessions for the sidebar."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM sessions ORDER BY created_at DESC")
    sessions = [dict(row) for row in c.fetchall()]
    conn.close()
    return sessions

def get_session_history(session_id):
    """Get full chat history for a specific session."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    messages = [dict(row) for row in c.fetchall()]
    conn.close()
    return messages

def learn_from_conversation(user_query: str, ai_response: str):
    """
    Background Task: Future integration point to save useful facts to Qdrant.
    """
    try:
        # Only memorize substantial answers
        if len(ai_response) > 50: 
            # print(f"🧠 [Background] Memorizing fact...")
            # rag = MedicalRAG(config)
            # if hasattr(rag, 'add_memory'):
            #     rag.add_memory(f"Q: {user_query}\nA: {ai_response}")
            pass
    except Exception as e:
        print(f"❌ Failed to learn: {e}")

# Set up directories
UPLOAD_FOLDER = "uploads/backend"
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"
SPEECH_DIR = "uploads/speech"

# Create directories if they don't exist
for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SPEECH_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mount static files
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Initialize ElevenLabs client
client = ElevenLabs(
    api_key=config.speech.eleven_labs_api_key,
)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_audio():
    """Deletes all .mp3 files in the uploads/speech folder every 5 minutes."""
    while True:
        try:
            files = glob.glob(f"{SPEECH_DIR}/*.mp3")
            for file in files:
                os.remove(file)
            print("Cleaned up old speech files.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        time.sleep(300)

# Start background cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
cleanup_thread.start()

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    conversation_history: List = Field(default_factory=list)



class SpeechRequest(BaseModel):
    text: str
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# --- SESSION ENDPOINTS ---

@app.get("/sessions")
def get_sessions_route():
    return {"sessions": get_user_sessions()}

@app.get("/sessions/{session_id}")
def load_session_route(session_id: str):
    return {"history": get_session_history(session_id)}

@app.post("/new-chat")
def start_new_chat():
    session_id = create_new_session()
    return {"session_id": session_id}

@app.post("/chat")
def chat(
    request: QueryRequest, 
    background_tasks: BackgroundTasks,
    response: Response,
    session_id: Optional[str] = Cookie(None)
):
    # Logic to get or create session
    active_session_id = request.session_id or session_id
    if not active_session_id or active_session_id == "null":
        active_session_id = create_new_session()

    try:
        # 1. Save User Message
        save_message(active_session_id, "user", request.query)

        # --- NEW: Dynamic Title Renaming ---
        # Check if this session is still named "New Chat" (meaning it's fresh)
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT title FROM sessions WHERE id = ?", (active_session_id,))
        row = c.fetchone()
        
        # If title is default, rename it to the user's query (truncated)
        if row and row[0] == "New Chat":
            new_title = request.query[:30] + "..." if len(request.query) > 30 else request.query
            c.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, active_session_id))
            conn.commit()
        conn.close()
        # -----------------------------------

        # 2. Process AI Response
        response_data = process_query(request.query)
        response_text = response_data['messages'][-1].content
        
        # 3. Save AI Message
        save_message(active_session_id, "assistant", response_text)

        # 4. Trigger Learning (Background)
        if response_data.get("agent_name") in ["RAG_AGENT", "PLANT_DISEASE_AGENT", "INSECT_AGENT"]:
            background_tasks.add_task(learn_from_conversation, request.query, response_text)

        response.set_cookie(key="session_id", value=active_session_id)

        return {
            "status": "success",
            "response": response_text, 
            "agent": response_data.get("agent_name", "Unknown Agent"),
            "session_id": active_session_id,
            "title": row[0] if row else "New Chat" # Return title so frontend can update if needed
        }
    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_image(
    background_tasks: BackgroundTasks,
    response: Response,
    image: UploadFile = File(...), 
    text: str = Form(""),
    session_id: Optional[str] = Cookie(None)
):
    if not allowed_file(image.filename):
        return JSONResponse(status_code=400, content={"status": "error", "response": "Unsupported file type."})
    
    # Session Handling
    if not session_id or session_id == "null":
        session_id = create_new_session(title="Image Analysis")

    # Save File
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file_content = await image.read()
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    try:
        # 1. Process
        query = {"text": text, "image": file_path}
        response_data = process_query(query)
        response_text = response_data['messages'][-1].content

        # 2. Save History
        display_text = text if text else f"[Uploaded Image: {image.filename}]"
        save_message(session_id, "user", display_text)
        save_message(session_id, "assistant", response_text)

        # 3. Learning
        background_tasks.add_task(learn_from_conversation, "User uploaded an image", response_text)

        response.set_cookie(key="session_id", value=session_id)

        return {
            "status": "success",
            "response": response_text, 
            "agent": response_data.get("agent_name", "Unknown Agent"),
            "uploaded_file": file_path,
            "session_id": session_id
        }
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
def validate_output(
    response: Response,
    validation_result: str = Form(...), 
    comments: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None)
):
    """Handle human validation (Agronomist/Farmer check)."""
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        response.set_cookie(key="session_id", value=session_id)
        
        validation_query = f"Validation result: {validation_result}"
        if comments:
            validation_query += f" Comments: {comments}"
        
        response_data = process_query(validation_query)
        response_text = response_data['messages'][-1].content

        status = "validated" if validation_result.lower() == 'yes' else "rejected"
        message = "**Output confirmed by user**" if status == "validated" else "**Output rejected by user**"

        return {
            "status": status,
            "message": message,
            "response": response_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe speech using ElevenLabs."""
    if not audio.filename:
        return JSONResponse(status_code=400, content={"error": "No audio file"})
    
    try:
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.webm"
        
        audio_content = await audio.read()
        with open(temp_audio, "wb") as f:
            f.write(audio_content)
        
        # Convert to MP3
        mp3_path = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.mp3"
        try:
            audio = AudioSegment.from_file(temp_audio)
            audio.export(mp3_path, format="mp3")
            
            with open(mp3_path, "rb") as mp3_file:
                audio_data = mp3_file.read()

            transcription = client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng",
                diarize=True,
            )
            
            # Clean up
            if os.path.exists(temp_audio): os.remove(temp_audio)
            if os.path.exists(mp3_path): os.remove(mp3_path)
            
            return {"transcript": transcription.text}

        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Processing error: {str(e)}"})
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    """Generate speech using ElevenLabs."""
    try:
        if not request.text:
            return JSONResponse(status_code=400, content={"error": "Text is required"})
        
        elevenlabs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{request.voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": config.speech.eleven_labs_api_key
        }
        payload = {
            "text": request.text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }

        response = requests.post(elevenlabs_url, headers=headers, json=payload)

        if response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": f"ElevenLabs Error: {response.text}"})
        
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio_path = f"./{SPEECH_DIR}/{uuid.uuid4()}.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(response.content)

        return FileResponse(path=temp_audio_path, media_type="audio/mpeg", filename="generated_speech.mp3")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(
        status_code=413,
        content={"status": "error", "response": f"File too large. Max: {config.api.max_image_upload_size}MB"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main_assistant:app",      # 🔑 import string (THIS FIXES IT)
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["."]
    )