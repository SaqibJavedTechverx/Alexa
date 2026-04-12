# Alexa Educational Voice Assistant

This project is a simple voice-enabled education assistant built around Alexa-style interaction. The backend is powered by Flask, and the frontend uses plain HTML, CSS, and JavaScript (no Streamlit).

## Features

- Voice input using the Web Speech API
- Text-to-speech output via gTTS (Amazon Alexa style)
- Interaction history and basic analytics displayed in the browser
- Backend exposes endpoints for `/ask`, `/history/<user_id>`, `/tts` and `/reports/<user_id>`

## Setup

1. **Clone the repository** (or place files in a directory).
2. **Create a virtual environment** and activate it:
   ```powershell
   python -m venv venv
   & venv\Scripts\Activate.ps1  # on Windows PowerShell
   # or source venv/bin/activate    # on macOS/Linux
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment variables**
   Create a `.env` file in the project root.

   **Web voice UI → same backend as your Alexa skill (recommended):**
   ```ini
   ALEXA_LAMBDA_ARN=arn:aws:lambda:eu-west-2:123456789012:function:YourSkillFunction
   AWS_REGION=eu-west-2
   ```
   Configure AWS credentials on this machine (`aws configure` or environment variables) with **`lambda:InvokeFunction`** on that function. In **Lambda → Configuration → Permissions**, add a **resource-based policy** (or “AWS account” invoke permission) so your IAM user can invoke the function—not only Alexa’s service.

   **Optional:** local dev without Lambda — set `ASK_USE_GEMINI=1` and `GOOGLE_API_KEY` to answer via Gemini from Flask instead (not used when `ALEXA_LAMBDA_ARN` is set).

   **Alexa skill (Lambda):** `alexa_lambda/lambda_function.py` answers with **DuckDuckGo instant answers** by default (no Gemini). To use Gemini on Lambda, set `USE_GEMINI=1` and `GOOGLE_API_KEY` there. Upload `alexa_lambda_deploy.zip` from `.\scripts\package_lambda.ps1`. Handler: `lambda_function.lambda_handler`.

   Also install **gTTS** for `/tts` (see `requirements.txt`).

5. **Run the backend**:
   ```bash
   python app.py
   ```
   On first run the app will create a local SQLite database file (`data.db`) to
   store interactions. You can change the database URL by setting
   `DATABASE_URL` in your `.env` (e.g. `postgresql://user:pass@host/db`).

6. **Open the frontend**
   Navigate to `http://localhost:5000` which will redirect you to the voice
   assistant page. The project now uses separate templates for each section:
   - `/alexa` – voice assistant and realtime stats
   - `/history` – tabular interaction history with chart
   - `/reports` – activity metrics and visualizations
   Each page retains the common sidebar for navigation.


### Exporting conversations as PDF

The backend now supports download of a user transcript: make a request to
`/export/<user_id>` and a PDF file will be returned containing each question and
answer with timestamps. This uses the `fpdf` package (included above).

Example:
```bash
curl -o student_noman.pdf http://localhost:5000/export/student_noman
```
## Project Structure

```
app.py                 # Flask backend
requirements.txt       # Python dependencies
templates/
  └─ index.html        # Frontend HTML, CSS, and JS
streamlit_app.py       # Deprecated streamlit frontend (for reference)
interaction_model.json # Alexa skill model (if you deploy to Alexa)
```

## Notes

- The frontend uses the browser **SpeechRecognition** API (Chrome/Edge) and **`/tts`** (gTTS) for playback.
- **`/ask`** calls your **Alexa skill Lambda** when **`ALEXA_LAMBDA_ARN`** is set (same logic as Echo). Otherwise you can use **`ASK_USE_GEMINI=1`** + **`GOOGLE_API_KEY`** for local-only testing.
- The skill Lambda uses **DuckDuckGo instant answers** by default; set **`USE_GEMINI=1`** and **`GOOGLE_API_KEY`** on Lambda only if you want Gemini there.

Feel free to modify and enhance the assistant for your educational needs!