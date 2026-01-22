# 🌾 Buddy-in-Fields: AI Farmer Assistant

**Buddy-in-Fields** is a multimodal AI agent designed to assist farmers and agricultural experts. It combines Retrieval-Augmented Generation (RAG) with advanced audio and image processing capabilities to answer queries based on uploaded documents, analyze plant images, and interact via voice.

---

## 🚀 Key Features

* **Multimodal Interaction**:
    * **Text**: Chat with the assistant using natural language.
    * **Voice**: Speech-to-Text (STT) and Text-to-Speech (TTS) using ElevenLabs and OpenAI.
    * **Vision**: Upload images of plants/documents. The system uses OCR and Vision models to analyze content.
* **RAG (Retrieval Augmented Generation)**:
    * Ingests PDF, DOCX, TXT, and Markdown files.
    * Uses **Qdrant** vector database for efficient semantic search.
* **Intelligent Agent**: Built with **LangGraph** to manage conversation state and tool usage.
* **Session Management**: Persists chat history for continuous conversations.
* **Dockerized**: Fully containerized for easy deployment and reproducibility.

---

## 🛠️ Tech Stack

* **Backend**: Python, FastAPI
* **Orchestration**: LangChain, LangGraph
* **LLMs**: Google Gemini (via LangChain), OpenAI (optional)
* **Vector DB**: Qdrant (Local mode)
* **Audio**: ElevenLabs, FFmpeg, PyDub
* **Vision**: OpenCV, EasyOCR, Pillow
* **Deployment**: Docker, Docker Compose

---

## 📋 Prerequisites

Before running the application, ensure you have the following installed:

1.  **Docker Desktop** (or Docker Engine + Compose plugin)
2.  **API Keys**: You will need keys for the following services:
    * `GOOGLE_GENAI_API_KEY` (For the LLM)
    * `ELEVEN_LABS_API_KEY` (For Voice output)
    * `TAVILY_API_KEY` (For web search capabilities)

---

## 🐳 Quick Start with Docker (Recommended)

This project is optimized for Docker to ensure reproducibility across different machines.

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/buddy-in-fields.git](https://github.com/yourusername/buddy-in-fields.git)
cd buddy-in-fields
```

### 2. Configure Environment Variables
Create a .env file in the root directory. You can copy the structure below:

```bash

# Create the file
touch .env
```
Paste the following into your .env file:


```bash
# LLM Providers
google_genai_api_key=your_google_api_key_here

# Voice Services
ELEVEN_LABS_API_KEY=your_elevenlabs_key_here

# Search Tools
TAVILY_API_KEY=your_tavily_key_here

# Optional: Qdrant (Leave blank to use local storage inside Docker)
QDRANT_URL=
QDRANT_API_KEY=
```

### 3. Build and Run
Execute the following command to build the image and start the container:

```bash
docker-compose up --build
```
*Note: The first build may take a few minutes as it downloads PyTorch and other machine learning dependencies.*

### 4. Access the Application
Once the logs show Application startup complete, the API is live at:

- API Root: `http://localhost:8000`

- Swagger Documentation: `http://localhost:8000/docs` (Use this to test endpoints)

## 💾 Data Persistence
The `docker-compose.yml` is configured to ensure your data survives container restarts.

- `./data`: Maps to the container's Qdrant database and processed documents.

- `./uploads`: Maps to user uploaded images and audio files.

- `./.env`: Maps your secrets securely.

To reset the database entirely, stop the container and delete the local data folder:

```bash
docker-compose down
rm -rf data/
```

## 🔌 API Endpoints Overview
Chat
- `POST /chat`: Standard text-based chat.

- `POST /chat/audio`: Upload an audio file (wav/mp3) to query the assistant.

- `POST /chat/image`: Upload an image to analyze visual data.

Knowledge Base
- `POST /upload_doc`: Upload a PDF/Document to add it to the RAG knowledge base.

Utilities
- `GET /get_audio/{filename}`: Retrieve generated audio responses.

- `GET /get_image/{filename}`: Retrieve processed images.

## 🔧 Local Development (Without Docker)
If you prefer to run it locally for debugging:

Create Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install System Dependencies:

- You must have `ffmpeg` installed on your system.

- You need C++ build tools for some Python libraries.

Install Python Packages:

```bash
pip install -r requirements.txt
```
Run:

```bash
uvicorn main_assistant:app --reload
```
## 🐞 Troubleshooting

### 1. ImportError: libGL.so.1 or libsm6

**Cause:**  
Missing system dependencies for OpenCV.

**Fix:**  
Ensure you are using the provided Dockerfile. It already installs:
- `libgl1-mesa-glx`
- `libsm6`

---

### 2. Docker Memory Error (Exit Code 137)

**Cause:**  
Machine learning models require sufficient RAM.

**Fix:**  
Increase Docker memory to **at least 4GB**:

`Preferences → Resources → Memory`

---

### 3. Audio not processing

**Cause:**  
`ffmpeg` not found.

**Fix:**  
- Docker users: The Dockerfile installs ffmpeg automatically.  
- Local users: Install manually:

**Ubuntu/Debian**
```bash
sudo apt install ffmpeg
```
**Mac**
```bash
brew install ffmpeg
```

**Windows**
```bash
choco install ffmpeg
```
