# app.py
from __future__ import annotations

from flask import Flask, request, jsonify, send_file, render_template, make_response, redirect, url_for, flash, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user # type: ignore
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
import ast
import difflib
import json
import operator as opmod
import os
import re
import io
import secrets
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from fpdf import FPDF  # for PDF export
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

# ── Load environment variables ────────────────────────────────
load_dotenv()

# Strip quotes/spaces from AWS vars — stray quotes cause UnrecognizedClientException.
for _k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
    _v = os.getenv(_k)
    if _v:
        os.environ[_k] = _v.strip().strip('"').strip("'")

# Using static AWS keys from .env on a laptop: disable EC2 instance metadata lookup.
# Without this, boto3 can hang a long time trying 169.254.169.254 before using your keys.
if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_EC2_METADATA_DISABLED") is None:
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

# Optional Gemini: set ASK_USE_GEMINI=1 and GOOGLE_API_KEY. Use google.generativeai
# GenerativeModel (not genai.chat.create / OpenAI-style APIs).
# ASK_GEMINI_FIRST=1 (with ALEXA_LAMBDA_ARN set): answer general questions with Gemini;
# route skill-style utterances (e.g. "who are you") to the Alexa Lambda via AssistantIdentityIntent.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
ALEXA_LAMBDA_ARN = os.getenv("ALEXA_LAMBDA_ARN", "").strip()
genai = None
if GOOGLE_API_KEY and os.getenv("ASK_USE_GEMINI", "").strip().lower() in ("1", "true", "yes"):
    try:
        import google.generativeai as _genai
        genai = _genai
        genai.configure(api_key=GOOGLE_API_KEY)
    except ImportError:
        print("google-generativeai package not installed, skipping Gemini fallback")


def _voice_ask_gemini_first() -> bool:
    """
    Gemini for general Q&A; Alexa Lambda only for skill-specific (custom) intents.
    Requires ASK_USE_GEMINI, GOOGLE_API_KEY, ALEXA_LAMBDA_ARN, and ASK_GEMINI_FIRST=1.
    """
    if not (genai and GOOGLE_API_KEY and ALEXA_LAMBDA_ARN):
        return False
    if os.getenv("ASK_USE_GEMINI", "").strip().lower() not in ("1", "true", "yes"):
        return False
    return os.getenv("ASK_GEMINI_FIRST", "").strip().lower() in ("1", "true", "yes")


def _voice_ask_prefers_gemini() -> bool:
    """If True, /ask uses Gemini for everything (including identity), even when Lambda is set."""
    if _voice_ask_gemini_first():
        return False
    return (
        bool(genai and GOOGLE_API_KEY)
        and os.getenv("ASK_USE_GEMINI", "").strip().lower() in ("1", "true", "yes")
        and os.getenv("VOICE_ASK_USE_GEMINI", "").strip().lower() in ("1", "true", "yes")
    )


# Keep in sync with alexa_lambda/lambda_function.py ASSISTANT_IDENTITY_SPEECH
ASSISTANT_IDENTITY_SPEECH = (
    "I'm Alexa. I'm your Alexa Education study assistant. Ask me any school topic."
)

# Keep in sync with alexa_lambda/lambda_function.py UNCLEAR_UTTERANCE_SPEECH
UNCLEAR_UTTERANCE_SPEECH = (
    "I didn't catch that. Ask again in one clear sentence about a school topic, "
    "for example: what is photosynthesis, or what is one hundred divided by five."
)

_MATH_PREFIX = re.compile(
    r"^(what\'s|what is|whats|calculate|compute|solve)\s+",
    re.I,
)
_MATH_ALLOWED_CHARS = re.compile(r"^[\d\s+\-*/().,x×÷^%]+$", re.I)

_ALLOWED_BINOPS = {
    ast.Add: opmod.add,
    ast.Sub: opmod.sub,
    ast.Mult: opmod.mul,
    ast.Div: opmod.truediv,
    ast.Pow: opmod.pow,
    ast.Mod: opmod.mod,
}


def _looks_like_arithmetic_expression(q: str) -> bool:
    """True for bare expressions like '3 + 3 x 2' or 'what is 12 / 4' (digits + operators only)."""
    s = (q or "").strip()
    if not s:
        return False
    tail = _MATH_PREFIX.sub("", s).strip()
    if not re.search(r"\d", tail):
        return False
    return bool(_MATH_ALLOWED_CHARS.fullmatch(tail))


def _normalize_math_expression(s: str) -> str:
    s = (s or "").strip()
    s = _MATH_PREFIX.sub("", s).strip()
    s = s.replace("×", "*").replace("÷", "/")
    s = re.sub(r"(?<=\d)\s*[x×]\s*(?=\d)", "*", s)
    s = re.sub(r"(?<=\d)[x×](?=\d)", "*", s, flags=re.I)
    s = s.replace("^", "**")
    return " ".join(s.split())


def _safe_eval_math_ast(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _safe_eval_math_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("bool")
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("constant")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _safe_eval_math_ast(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval_math_ast(node.left)
        right = _safe_eval_math_ast(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    raise ValueError("unsupported")


def answer_spoken_arithmetic(question: str) -> str | None:
    """Evaluate a short arithmetic expression with or without 'what is'. Returns None if not math."""
    if not _looks_like_arithmetic_expression(question):
        return None
    expr = _normalize_math_expression(question)
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        v = _safe_eval_math_ast(tree)
        if isinstance(v, float):
            if abs(v) < 1e15 and abs(v - round(v)) < 1e-9:
                v = int(round(v))
        if isinstance(v, float):
            spoken = f"{v:.10g}".rstrip("0").rstrip(".")
        else:
            spoken = str(v)
        return f"The answer is {spoken}."
    except Exception:
        return None


def _web_utterance_unclear(question: str) -> bool:
    """
    Fragmentary speech-to-text, mostly numbers, or non-questions often hit random
    web snippets (e.g. '50064 from 500' → unrelated facts). Ask the user to repeat.
    """
    q = (question or "").strip()
    if len(q) < 2:
        return True
    ql = q.lower()

    if _looks_like_arithmetic_expression(q):
        return False

    _clear = (
        "what is ",
        "what are ",
        "what was ",
        "what were ",
        "what does ",
        "what did ",
        "what do ",
        "what's ",
        "whats ",
        "who is ",
        "who was ",
        "who are ",
        "when did ",
        "when was ",
        "when is ",
        "where is ",
        "where are ",
        "where was ",
        "why ",
        "how ",
        "explain ",
        "define ",
        "describe ",
        "compare ",
        "tell me about ",
        "tell me what ",
        "calculate ",
        "can you explain",
    )
    if any(ql.startswith(p) for p in _clear) and len(ql) >= 12:
        return False

    if re.fullmatch(r"[\d\s,.\-]+", ql):
        return True
    if re.fullmatch(
        r"\d{1,12}\s+(from|to|of|and|or|in|on|by|over|into|out|off)\s+\d{1,12}\s*",
        ql,
        re.I,
    ):
        return True

    words = [w for w in re.split(r"\s+", ql) if w]
    if not words:
        return True
    if len(words) <= 4:
        numish = sum(1 for w in words if re.fullmatch(r"\d[\d,.\-]*", w))
        letter_words = sum(1 for w in words if re.search(r"[a-z]", w))
        if numish >= 2 and letter_words <= 1:
            return True

    letters = len(re.findall(r"[a-z]", ql))
    digits = len(re.findall(r"\d", ql))
    if len(ql) <= 24 and digits >= 4 and letters <= 12:
        return True

    return False


def _web_utterance_matches_assistant_identity(question: str) -> bool:
    """Web UI has no Alexa NLU — use heuristics to invoke AssistantIdentityIntent like the skill."""
    q = (question or "").lower().strip()
    if not q:
        return False
    needles = (
        "which assistant",
        "what assistant",
        "who are you",
        "what are you",
        "your name",
        "what's your name",
        "whats your name",
        "who is this",
        "what should i call you",
        "do you have a name",
        "are you alexa",
        "are you siri",
        "are you google",
        "are you chatgpt",
        "what ai are you",
        "what assistant is this",
        "are you an ai",
        "are you artificial",
    )
    return any(n in q for n in needles)


def generate_education_answer(question: str) -> str:
    """Short, speakable explanation for students."""
    if not genai or not GOOGLE_API_KEY:
        return f"I heard: {question}. Add GOOGLE_API_KEY and google-generativeai for full answers."
    arith = answer_spoken_arithmetic(question)
    if arith is not None:
        return arith
    if _web_utterance_unclear(question):
        return UNCLEAR_UTTERANCE_SPEECH
    try:
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=(
                "You are a friendly tutor. Answer clearly and accurately. "
                "Use plain language suitable for text-to-speech. No markdown or bullet symbols. "
                "Keep answers concise (roughly under 150 words) unless a short list helps. "
                "If the message is gibberish or not a real school question, say you did not catch "
                "it and ask them to repeat in one clear sentence."
            ),
        )
        # Hard timeout so the browser is not stuck on "THINKING" forever if the API stalls.
        timeout_sec = float(os.getenv("GEMINI_TIMEOUT_SEC", "30"))
        r = model.generate_content(
            question.strip(),
            request_options={"timeout": timeout_sec},
        )
        try:
            text = r.text
        except ValueError as ve:
            # Blocked prompt, safety, or empty parts — .text raises
            return f"I could not answer that safely or the response was blocked. {ve!s}"
        if not (text or "").strip():
            return "I got an empty response. Try a different question or check GEMINI_MODEL in your .env file."
        return (text or "").strip()
    except Exception as e:
        return f"Sorry, I couldn't answer that right now. ({e})"


def _region_from_lambda_arn(arn: str) -> str:
    parts = arn.split(":")
    if len(parts) > 3 and parts[2] == "lambda" and parts[3]:
        return parts[3]
    return os.getenv("AWS_REGION", "us-east-1")


def _synthetic_alexa_intent_event(question: str) -> dict:
    """Same shape the Alexa skill Lambda expects for AskQuestionIntents."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "version": "1.0",
        "session": {
            "new": False,
            "sessionId": "flask-web",
            "application": {"applicationId": "amzn1.ask.skill.flask-bridge"},
            "user": {"userId": "flask-user"},
        },
        "context": {
            "System": {
                "application": {"applicationId": "amzn1.ask.skill.flask-bridge"},
                "user": {"userId": "flask-user"},
            }
        },
        "request": {
            "type": "IntentRequest",
            "requestId": "flask-ask-bridge",
            "timestamp": ts,
            "locale": "en-US",
            "intent": {
                "name": "AskQuestionIntents",
                "slots": {
                    "question": {"name": "question", "value": question},
                },
            },
        },
    }


def _synthetic_assistant_identity_intent_event() -> dict:
    """Mirrors skill model AssistantIdentityIntent (samples live in interaction_model.json)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "version": "1.0",
        "session": {
            "new": False,
            "sessionId": "flask-web",
            "application": {"applicationId": "amzn1.ask.skill.flask-bridge"},
            "user": {"userId": "flask-user"},
        },
        "context": {
            "System": {
                "application": {"applicationId": "amzn1.ask.skill.flask-bridge"},
                "user": {"userId": "flask-user"},
            }
        },
        "request": {
            "type": "IntentRequest",
            "requestId": "flask-identity-bridge",
            "timestamp": ts,
            "locale": "en-US",
            "intent": {
                "name": "AssistantIdentityIntent",
                "slots": {},
            },
        },
    }


def _lambda_invoke_raw(arn: str, payload_bytes: bytes) -> dict:
    """Sync invoke; used inside a thread with a hard timeout."""
    import boto3
    from botocore.config import Config

    cfg = Config(
        connect_timeout=5,
        read_timeout=int(os.getenv("LAMBDA_INVOKE_READ_TIMEOUT", "22")),
        retries={"max_attempts": 1, "mode": "standard"},
    )
    client = boto3.client(
        "lambda",
        region_name=_region_from_lambda_arn(arn),
        config=cfg,
    )
    return client.invoke(
        FunctionName=arn,
        InvocationType="RequestResponse",
        Payload=payload_bytes,
    )


def _invoke_alexa_skill_lambda(event: dict, log_hint: str) -> str:
    """Shared invoke + response parse for AskQuestionIntents vs AssistantIdentityIntent."""
    if not ALEXA_LAMBDA_ARN:
        return (
            "Set ALEXA_LAMBDA_ARN in your .env to your skill's Lambda ARN, "
            "and configure AWS credentials (for example aws configure) with permission to invoke that function."
        )

    app.logger.info("Invoking Alexa skill Lambda (%s)", log_hint)
    payload_bytes = json.dumps(event).encode("utf-8")
    outer_timeout = float(os.getenv("LAMBDA_INVOKE_TOTAL_TIMEOUT", "28"))

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_lambda_invoke_raw, ALEXA_LAMBDA_ARN, payload_bytes)
            resp = fut.result(timeout=outer_timeout)
    except FuturesTimeoutError:
        return (
            "Calling your Alexa skill timed out. Check Lambda logs, cold start, and "
            "LAMBDA_INVOKE_READ_TIMEOUT / LAMBDA_INVOKE_TOTAL_TIMEOUT in .env."
        )
    except ModuleNotFoundError:
        return "Install boto3: pip install boto3"
    except Exception as e:
        return (
            f"Could not invoke your Alexa skill Lambda. Check ARN, region, and IAM permissions. ({e})"
        )

    raw = resp["Payload"].read()
    try:
        body = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return f"Unexpected Lambda response: {raw[:500]!r}"

    if resp.get("FunctionError"):
        return f"Lambda error: {body.get('errorMessage', body)}"

    try:
        text = body["response"]["outputSpeech"]["text"]
        app.logger.info("Lambda returned answer length=%s", len(text or ""))
        return text
    except (KeyError, TypeError):
        return str(body)[:2000]


def answer_via_alexa_skill(question: str) -> str:
    """AskQuestionIntents → Lambda (same as device skill)."""
    q = (question[:80] + "…") if len(question) > 80 else question
    return _invoke_alexa_skill_lambda(
        _synthetic_alexa_intent_event(question),
        f"AskQuestionIntents: {q!r}",
    )


def answer_via_alexa_skill_identity() -> str:
    """AssistantIdentityIntent → Lambda (identity is modeled in the skill, not parsed in AskQuestion)."""
    return _invoke_alexa_skill_lambda(
        _synthetic_assistant_identity_intent_event(),
        "AssistantIdentityIntent",
    )


def voice_ask_backend() -> str:
    """Which backend /ask uses: for logs and JSON."""
    if _voice_ask_gemini_first():
        return "gemini_first"
    if _voice_ask_prefers_gemini():
        return "gemini_fallback"
    if ALEXA_LAMBDA_ARN:
        return "alexa_skill"
    if genai and GOOGLE_API_KEY:
        return "gemini_fallback"
    return "missing_config"


def ask_backend_label(code: str) -> str:
    """Short human-readable line for UI and API."""
    return {
        "alexa_skill": "Answers: Alexa skill (Lambda)",
        "gemini_first": "Answers: Gemini first; Alexa skill for custom intents",
        "gemini_fallback": "Answers: Gemini only (no Lambda ARN)",
        "missing_config": "Answers: not configured (set Lambda or ASK_USE_GEMINI)",
        "quiz_mode": "Answers: your quiz (local)",
    }.get(code, code)


def answer_for_voice_ui(question: str) -> str:
    """
    Routing:
    - ASK_GEMINI_FIRST=1 + Lambda + Gemini: Gemini for open Q&A; Lambda for skill intents (e.g. identity).
    - VOICE_ASK_USE_GEMINI=1: all Gemini when Lambda also set (legacy).
    - Else: Lambda if set, else Gemini if configured.
    """
    bk = voice_ask_backend()
    app.logger.info("/ask backend=%s", bk)

    if _voice_ask_gemini_first():
        if _web_utterance_matches_assistant_identity(question):
            return answer_via_alexa_skill_identity()
        arith = answer_spoken_arithmetic(question)
        if arith is not None:
            return arith
        if _web_utterance_unclear(question):
            return UNCLEAR_UTTERANCE_SPEECH
        return generate_education_answer(question)

    if _web_utterance_matches_assistant_identity(question):
        if ALEXA_LAMBDA_ARN and not _voice_ask_prefers_gemini():
            return answer_via_alexa_skill_identity()
        return ASSISTANT_IDENTITY_SPEECH
    arith = answer_spoken_arithmetic(question)
    if arith is not None:
        return arith
    if _web_utterance_unclear(question):
        return UNCLEAR_UTTERANCE_SPEECH
    if _voice_ask_prefers_gemini():
        return generate_education_answer(question)
    if ALEXA_LAMBDA_ARN:
        return answer_via_alexa_skill(question)
    if genai and GOOGLE_API_KEY:
        return generate_education_answer(question)
    if bk == "missing_config":
        app.logger.warning(
            "/ask missing both ALEXA_LAMBDA_ARN and Gemini; set ASK_USE_GEMINI=1 + GOOGLE_API_KEY for local dev"
        )
    return (
        "Add ALEXA_LAMBDA_ARN (and AWS credentials) to use your Alexa skill from the web app. "
        "For local-only testing without Lambda, set ASK_USE_GEMINI=1 and GOOGLE_API_KEY."
    )


# Backend Flask app (serves HTML/JS frontend)
app = Flask(__name__, static_folder='static', template_folder='templates')
# allow cross-origin requests (frontend and backend are same origin but useful during dev)
CORS(app)

# Database configuration (SQLite locally; set DATABASE_URL for AWS RDS PostgreSQL/MySQL)
def _normalize_database_url(uri: str) -> str:
    """Heroku-style postgres:// → postgresql:// for SQLAlchemy + psycopg2."""
    u = (uri or "").strip()
    if u.startswith("postgres://"):
        return "postgresql://" + u[len("postgres://") :]
    return u


def _postgres_apply_ssl_defaults(uri: str) -> str:
    """Append sslmode for Amazon RDS hosts unless already set or DATABASE_SSLMODE=disable."""
    u = _normalize_database_url((uri or "").strip())
    if not u.startswith("postgresql://"):
        return u
    parsed = urlparse(u)
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if "sslmode" in q:
        return u
    mode = os.getenv("DATABASE_SSLMODE", "").strip().lower()
    if mode in ("disable", "false", "0", "no"):
        return u
    if mode:
        q["sslmode"] = mode
        return urlunparse(parsed._replace(query=urlencode(q)))
    host = (parsed.hostname or "").lower()
    if "rds.amazonaws.com" in host:
        q["sslmode"] = "require"
        return urlunparse(parsed._replace(query=urlencode(q)))
    return u


def _database_url_for_log(uri: str) -> str:
    """Host + path + ssl hint only — never log credentials."""
    u = _normalize_database_url((uri or "").strip())
    if u.startswith("sqlite"):
        return u.split("?", 1)[0]
    try:
        p = urlparse(u)
        q = dict(parse_qsl(p.query, keep_blank_values=True))
        ssl_hint = q.get("sslmode", "")
        tail = f" sslmode={ssl_hint}" if ssl_hint else ""
        port = f":{p.port}" if p.port else ""
        return f"{p.scheme}://{p.hostname or '?'}{port}{p.path or ''}{tail}"
    except Exception:
        return "(configured)"


db_url = _postgres_apply_ssl_defaults(os.getenv("DATABASE_URL", "sqlite:///data.db"))
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# RDS / managed DB: avoid stale connections; SQLite uses defaults
if db_url.startswith("sqlite"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {}
else:
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": int(os.getenv("SQLALCHEMY_POOL_RECYCLE", "280")),
    }
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Nginx / ALB terminate TLS and forward HTTP to Gunicorn — trust X-Forwarded-Proto for https URLs.
if os.getenv("BEHIND_HTTPS_PROXY", "").strip().lower() in ("1", "true", "yes"):
    from werkzeug.middleware.proxy_fix import ProxyFix

    app.wsgi_app = ProxyFix(
        app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1
    )

if os.getenv("SESSION_COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes"):
    app.config["SESSION_COOKIE_SECURE"] = True
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

app.logger.info("SQLAlchemy database: %s", _database_url_for_log(db_url))

# initialize database
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# In-memory history storage (legacy, kept empty)
history = []

# --- database model(s)
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to sessions
    sessions = db.relationship('ConversationSession', backref='user', lazy=True)

class ConversationSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to interactions
    interactions = db.relationship('Interaction', backref='session', lazy=True, cascade='all, delete-orphan')

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('conversation_session.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'question': self.question,
            'answer': self.answer,
            'timestamp': self.timestamp.isoformat()
        }


class Quiz(db.Model):
    """User-created quiz; `slug` is used with voice: start quiz [slug].
    Global quizzes (`is_global`) belong to the system user and are visible to everyone."""
    __tablename__ = "quiz"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    slug = db.Column(db.String(120), nullable=False)
    is_global = db.Column(db.Boolean, nullable=False, default=False)
    status = db.Column(db.String(32), nullable=False, default="not_started")
    last_score_correct = db.Column(db.Integer, nullable=True)
    last_score_wrong = db.Column(db.Integer, nullable=True)
    last_attempt_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    questions = db.relationship(
        "QuizQuestion",
        back_populates="quiz",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="QuizQuestion.sort_order",
    )
    attempts = db.relationship(
        "QuizAttempt",
        back_populates="quiz",
        lazy=True,
        cascade="all, delete-orphan",
    )

    __table_args__ = (db.UniqueConstraint("user_id", "slug", name="uq_quiz_user_slug"),)


class QuizQuestion(db.Model):
    __tablename__ = "quiz_question"
    id = db.Column(db.Integer, primary_key=True)
    quiz_id = db.Column(db.Integer, db.ForeignKey("quiz.id"), nullable=False)
    sort_order = db.Column(db.Integer, nullable=False, default=0)
    prompt = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    quiz = db.relationship("Quiz", back_populates="questions")


class QuizAttempt(db.Model):
    __tablename__ = "quiz_attempt"
    id = db.Column(db.Integer, primary_key=True)
    quiz_id = db.Column(db.Integer, db.ForeignKey("quiz.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    conversation_session_id = db.Column(db.Integer, db.ForeignKey("conversation_session.id"), nullable=True)
    status = db.Column(db.String(32), nullable=False, default="in_progress")
    current_index = db.Column(db.Integer, nullable=False, default=0)
    correct_count = db.Column(db.Integer, nullable=False, default=0)
    wrong_count = db.Column(db.Integer, nullable=False, default=0)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    quiz = db.relationship("Quiz", back_populates="attempts")


# Built-in quizzes for all accounts (stored under a non-login system user).
SYSTEM_QUIZ_USERNAME = "__system_quizzes__"
SYSTEM_QUIZ_EMAIL = "system-quizzes@local.invalid"
_GLOBAL_QUIZ_SEEDS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "Science basics",
        "science-basics",
        [
            ("What gas do plants take in from the air for photosynthesis?", "carbon dioxide"),
            ("Which planet is known as the Red Planet?", "Mars"),
            ("How many legs does a typical insect have?", "six"),
        ],
    ),
    (
        "Math warm-up",
        "math-warmup",
        [
            ("What is fifteen divided by three?", "five"),
            ("What whole number is the square root of sixty four?", "eight"),
            ("What is nine times six?", "fifty four"),
        ],
    ),
    (
        "Geography facts",
        "geography-facts",
        [
            ("What is the capital of Japan?", "Tokyo"),
            ("Which ocean lies between the Americas and Europe and Africa?", "Atlantic"),
            ("What is the largest country in the world by land area?", "Russia"),
        ],
    ),
]


def _get_system_quiz_user_id() -> int | None:
    u = User.query.filter_by(username=SYSTEM_QUIZ_USERNAME).first()
    return u.id if u else None


def _ensure_quiz_is_global_column() -> None:
    """Add quiz.is_global on existing databases (SQLite + PostgreSQL)."""
    from sqlalchemy import inspect, text

    insp = inspect(db.engine)
    if "quiz" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("quiz")}
    if "is_global" in cols:
        return
    uri = (app.config.get("SQLALCHEMY_DATABASE_URI") or "")
    with db.engine.begin() as conn:
        if uri.startswith("sqlite"):
            conn.execute(text("ALTER TABLE quiz ADD COLUMN is_global INTEGER NOT NULL DEFAULT 0"))
        else:
            conn.execute(text("ALTER TABLE quiz ADD COLUMN is_global BOOLEAN NOT NULL DEFAULT false"))


def _seed_global_quizzes() -> None:
    """Create system user (no login) and generic quizzes once."""
    u = User.query.filter_by(username=SYSTEM_QUIZ_USERNAME).first()
    if not u:
        u = User.query.filter_by(email=SYSTEM_QUIZ_EMAIL).first()
    if not u:
        u = User(
            username=SYSTEM_QUIZ_USERNAME,
            email=SYSTEM_QUIZ_EMAIL,
            password_hash=generate_password_hash(secrets.token_urlsafe(48)),
        )
        db.session.add(u)
        try:
            db.session.commit()
        except IntegrityError:
            # Another Gunicorn worker may have inserted the same row, or email/username
            # already exists from a prior partial migration.
            db.session.rollback()
            u = User.query.filter_by(username=SYSTEM_QUIZ_USERNAME).first()
            if not u:
                u = User.query.filter_by(email=SYSTEM_QUIZ_EMAIL).first()
    if u is None:
        raise RuntimeError("Could not create or load system quiz user")

    for title, slug, pairs in _GLOBAL_QUIZ_SEEDS:
        existing = Quiz.query.filter_by(user_id=u.id, slug=slug).first()
        if existing:
            if not existing.is_global:
                existing.is_global = True
                db.session.commit()
            continue
        qz = Quiz(
            user_id=u.id,
            title=title,
            slug=slug,
            is_global=True,
            status="not_started",
        )
        db.session.add(qz)
        db.session.flush()
        for i, (prompt, answer) in enumerate(pairs):
            db.session.add(
                QuizQuestion(quiz_id=qz.id, sort_order=i, prompt=prompt, answer=answer)
            )
        db.session.commit()


def _quiz_last_completed_stats(quiz_ids: list[int], uid: int) -> dict[int, tuple[int, int, datetime | None]]:
    """Most recent completed attempt per quiz for this user."""
    if not quiz_ids:
        return {}
    rows = (
        QuizAttempt.query.filter(
            QuizAttempt.quiz_id.in_(quiz_ids),
            QuizAttempt.user_id == uid,
            QuizAttempt.status == "completed",
        )
        .order_by(QuizAttempt.completed_at.desc())
        .all()
    )
    out: dict[int, tuple[int, int, datetime | None]] = {}
    for r in rows:
        if r.quiz_id not in out:
            out[r.quiz_id] = (r.correct_count, r.wrong_count, r.completed_at)
    return out


def _quiz_in_progress_ids(quiz_ids: list[int], uid: int) -> set[int]:
    if not quiz_ids:
        return set()
    rows = (
        QuizAttempt.query.filter(
            QuizAttempt.quiz_id.in_(quiz_ids),
            QuizAttempt.user_id == uid,
            QuizAttempt.status == "in_progress",
        )
        .all()
    )
    return {r.quiz_id for r in rows}


def _slugify_quiz_name(name: str) -> str:
    s = (name or "").lower().strip()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_]+", "-", s.strip())
    return s.strip("-")[:120]


def _answers_semantically_close(expected: str, user: str) -> bool:
    """True if the spoken reply is close enough to the stored answer for a quiz."""
    def norm(x: str) -> str:
        t = (x or "").lower()
        t = re.sub(r"[^\w\s]", " ", t)
        return " ".join(t.split())

    e = norm(expected)
    u = norm(user)
    if not e or not u:
        return False
    if e == u:
        return True
    if len(e) >= 3 and (e in u or u in e):
        return True
    ratio = difflib.SequenceMatcher(None, e, u).ratio()
    if ratio >= 0.72:
        return True
    ew = [w for w in e.split() if len(w) > 1]
    if ew and all(w in u for w in ew):
        return True
    return False


def _find_quiz_for_voice(user_id: int, raw_name: str):
    slug = _slugify_quiz_name(raw_name)
    if slug:
        q = Quiz.query.filter_by(user_id=user_id, slug=slug).first()
        if q:
            return q
        sid = _get_system_quiz_user_id()
        if sid is not None:
            q = Quiz.query.filter_by(user_id=sid, slug=slug, is_global=True).first()
            if q:
                return q
    needle = (raw_name or "").strip().lower()
    if not needle:
        return None
    for q in Quiz.query.filter_by(user_id=user_id).all():
        if q.title.lower() == needle:
            return q
        if needle == q.slug.replace("-", " "):
            return q
    sid = _get_system_quiz_user_id()
    if sid is not None:
        for q in Quiz.query.filter_by(user_id=sid, is_global=True).all():
            if q.title.lower() == needle:
                return q
            if needle == q.slug.replace("-", " "):
                return q
    return None


def _quiz_speak_question(item: QuizQuestion, num: int, total: int) -> str:
    return f"Question {num} of {total}. {item.prompt}"


def _quiz_abandon_in_progress(conv_session_id: int) -> None:
    att = QuizAttempt.query.filter_by(
        conversation_session_id=conv_session_id,
        status="in_progress",
    ).first()
    if att:
        att.status = "abandoned"
        att.completed_at = datetime.utcnow()
        att.conversation_session_id = None


def try_quiz_voice_reply(question: str, conv_session: ConversationSession) -> str | None:
    """
    Handle voice quiz start, answers, and stop. Returns spoken reply or None to use normal /ask.
    """
    q = (question or "").strip()
    ql = q.lower()

    if re.match(r"^(stop|end|cancel|quit)\s+quiz\s*$", ql) or ql in (
        "stop quiz",
        "end quiz",
        "cancel quiz",
        "quit quiz",
    ):
        att = QuizAttempt.query.filter_by(
            conversation_session_id=conv_session.id,
            status="in_progress",
        ).first()
        if att:
            _quiz_abandon_in_progress(conv_session.id)
            qz = db.session.get(Quiz, att.quiz_id)
            if qz and not qz.is_global:
                qz.status = "completed" if qz.last_attempt_at else "not_started"
            db.session.commit()
            return (
                "Quiz stopped. Say start quiz and the quiz name when you want to try again."
            )
        return None

    att = QuizAttempt.query.filter_by(
        conversation_session_id=conv_session.id,
        status="in_progress",
    ).first()
    if att:
        return _quiz_process_user_answer(q, att)

    m = re.match(r"^(?:start|begin)\s+(?:the\s+)?quiz\s+(.+)$", q, re.I)
    if not m:
        m = re.match(r"^quiz\s+(.+)$", q, re.I)
    if m:
        raw = m.group(1).strip().rstrip(".").strip()
        raw = re.sub(r"\s+quiz\s*$", "", raw, flags=re.I).strip()
        return _quiz_voice_start(raw, conv_session)

    return None


def _quiz_voice_start(raw_name: str, conv_session: ConversationSession) -> str:
    user_id = conv_session.user_id
    if not raw_name:
        return "Say which quiz to start, for example: start quiz biology."

    quiz = _find_quiz_for_voice(user_id, raw_name)
    if not quiz:
        return (
            f"I don't have a quiz named {raw_name}. Open the Quizzes page to see the exact name, "
            "or create a new quiz."
        )

    questions = (
        QuizQuestion.query.filter_by(quiz_id=quiz.id)
        .order_by(QuizQuestion.sort_order, QuizQuestion.id)
        .all()
    )
    if not questions:
        return (
            f"The quiz {quiz.title} has no questions yet. Add questions on the Quizzes page first."
        )

    _quiz_abandon_in_progress(conv_session.id)

    natt = QuizAttempt(
        quiz_id=quiz.id,
        user_id=user_id,
        conversation_session_id=conv_session.id,
        status="in_progress",
        current_index=0,
        correct_count=0,
        wrong_count=0,
    )
    db.session.add(natt)
    quiz.status = "in_progress"
    db.session.commit()

    q1 = questions[0]
    return (
        f"Starting {quiz.title}. "
        + _quiz_speak_question(q1, 1, len(questions))
    )


def _quiz_process_user_answer(user_text: str, att: QuizAttempt) -> str:
    quiz = db.session.get(Quiz, att.quiz_id)
    if not quiz:
        att.status = "abandoned"
        db.session.commit()
        return "That quiz no longer exists."

    questions = (
        QuizQuestion.query.filter_by(quiz_id=quiz.id)
        .order_by(QuizQuestion.sort_order, QuizQuestion.id)
        .all()
    )
    idx = att.current_index
    if idx >= len(questions):
        att.status = "completed"
        att.completed_at = datetime.utcnow()
        att.conversation_session_id = None
        db.session.commit()
        return "This quiz is already finished. Say start quiz and the name to try again."

    cq = questions[idx]
    ok = _answers_semantically_close(cq.answer, user_text)
    if ok:
        att.correct_count += 1
        feedback = "Correct. "
    else:
        att.wrong_count += 1
        feedback = f"Wrong. The answer was: {cq.answer}. "

    att.current_index += 1

    if att.current_index >= len(questions):
        att.status = "completed"
        att.completed_at = datetime.utcnow()
        att.conversation_session_id = None
        total = att.correct_count + att.wrong_count
        if not quiz.is_global:
            quiz.last_score_correct = att.correct_count
            quiz.last_score_wrong = att.wrong_count
            quiz.last_attempt_at = att.completed_at
            quiz.status = "completed"
        db.session.commit()
        return (
            feedback
            + f"Quiz complete. You got {att.correct_count} correct and {att.wrong_count} wrong "
            f"out of {total}. Say start quiz {quiz.slug.replace('-', ' ')} to try again anytime."
        )

    nq = questions[att.current_index]
    db.session.commit()
    return feedback + _quiz_speak_question(nq, att.current_index + 1, len(questions))


def _migrate_sqlite_schema() -> None:
    """Add missing columns when the DB file predates model changes (SQLite has no auto-migrate)."""
    uri = app.config.get("SQLALCHEMY_DATABASE_URI") or ""
    if not uri.startswith("sqlite:"):
        return
    from sqlalchemy import inspect, text

    insp = inspect(db.engine)
    if "interaction" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("interaction")}
    if "session_id" in cols:
        return
    with db.engine.begin() as conn:
        conn.execute(text("ALTER TABLE interaction ADD COLUMN session_id INTEGER"))
    with db.engine.begin() as conn:
        conn.execute(text("DELETE FROM interaction WHERE session_id IS NULL"))


# create tables if they don't exist
with app.app_context():
    db.create_all()
    _migrate_sqlite_schema()
    _ensure_quiz_is_global_column()
    _seed_global_quizzes()

# Startup: which /ask backend is configured (Flask terminal / WSGI logs)
_bk = voice_ask_backend()
_msg = {
    "alexa_skill": "ALEXA_LAMBDA_ARN set — /ask uses your Alexa skill Lambda",
    "gemini_first": "ASK_GEMINI_FIRST — Gemini for Q&A; Lambda for custom skill intents (e.g. identity)",
    "gemini_fallback": "ASK_USE_GEMINI + GOOGLE_API_KEY — /ask uses Gemini (no Lambda ARN)",
    "missing_config": "Neither Lambda nor Gemini configured — /ask will return setup instructions",
}.get(_bk, _bk)
app.logger.info("Voice /ask: %s (backend=%s)", _msg, _bk)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    """Redirect root to login if not authenticated, else to alexa."""
    if current_user.is_authenticated:
        return redirect(url_for('alexa'))
    return redirect(url_for('login'))

@app.route("/login", methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and user.username == SYSTEM_QUIZ_USERNAME:
            flash("Invalid username or password")
            return redirect(url_for("login"))
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('alexa'))
        flash('Invalid username or password')
    
    return render_template('login.html')

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    """Signup page."""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        un = (username or "").strip()
        if un == SYSTEM_QUIZ_USERNAME:
            flash("That username is reserved.")
            return redirect(url_for("signup"))
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('signup'))
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('alexa'))
    
    return render_template('signup.html')

@app.route("/logout")
@login_required
def logout():
    """Logout user."""
    logout_user()
    return redirect(url_for('login'))

# The trademark Echo “Alexa” assistant voice is not licensed for third‑party web apps.
# Skills on a real device use that pipeline; custom sites use Polly (often the same Polly
# voice IDs skills can choose in SSML) or browser TTS—never a literal clone of Alexa’s voice.
VOICE_LIMITATION_NOTE = (
    "Amazon does not license the Echo Alexa assistant voice for custom websites. "
    "For that voice, use this skill on an Alexa device. "
    "Here: set USE_POLLY_TTS=1 for Amazon Polly neural speech (skills can use these same Polly voices in SSML)."
)


def _speech_mode_hint() -> str:
    """Shown on Alexa page for how answers are spoken."""
    if _use_polly_tts():
        v = os.getenv("POLLY_VOICE_ID", "Joanna")
        return f"Speech: Amazon Polly ({v}, neural) — same Polly catalog Alexa skills may use in SSML"
    return "Speech: browser voice first, else Google gTTS — add USE_POLLY_TTS=1 + polly:SynthesizeSpeech for Polly"


@app.route("/alexa")
@login_required
def alexa():
    """Voice assistant interface."""
    _bk = voice_ask_backend()
    return render_template(
        "alexa.html",
        active="alexa",
        ask_backend=_bk,
        ask_backend_label=ask_backend_label(_bk),
        use_polly_voice=_use_polly_tts(),
        speech_mode_hint=_speech_mode_hint(),
        voice_limitation_note=VOICE_LIMITATION_NOTE,
    )

@app.route("/history")
@login_required
def history_page():
    """Interactive history listing."""
    return render_template('history.html', active='history')

@app.route("/reports")
@login_required
def reports_page():
    """Learning reports dashboard."""
    return render_template('reports.html', active='reports')


def _save_quiz_from_form(quiz: Quiz, title: str, slug: str, prompts: list, answers: list) -> None:
    quiz.title = title[:200]
    quiz.slug = slug[:120]
    quiz.updated_at = datetime.utcnow()
    pairs = [(p.strip(), a.strip()) for p, a in zip(prompts, answers) if p.strip() and a.strip()]
    for old in list(quiz.questions):
        db.session.delete(old)
    for i, (p, a) in enumerate(pairs):
        db.session.add(
            QuizQuestion(quiz_id=quiz.id, sort_order=i, prompt=p, answer=a)
        )


@app.route("/quizzes")
@login_required
def quizzes_list():
    sid = _get_system_quiz_user_id()
    own = (
        Quiz.query.options(selectinload(Quiz.questions))
        .filter_by(user_id=current_user.id)
        .order_by(Quiz.updated_at.desc())
        .all()
    )
    global_qs: list[Quiz] = []
    if sid is not None:
        global_qs = (
            Quiz.query.options(selectinload(Quiz.questions))
            .filter(Quiz.user_id == sid, Quiz.is_global.is_(True))
            .order_by(Quiz.title.asc())
            .all()
        )
    quizzes = global_qs + own
    qids = [q.id for q in quizzes]
    attempt_counts: dict[int, int] = {}
    if qids:
        rows = (
            db.session.query(QuizAttempt.quiz_id, func.count(QuizAttempt.id))
            .filter(
                QuizAttempt.quiz_id.in_(qids),
                QuizAttempt.user_id == current_user.id,
            )
            .group_by(QuizAttempt.quiz_id)
            .all()
        )
        attempt_counts = {int(r[0]): int(r[1]) for r in rows}
    last_stats = _quiz_last_completed_stats(qids, current_user.id)
    in_progress = _quiz_in_progress_ids(qids, current_user.id)
    return render_template(
        "quizzes.html",
        active="quizzes",
        quizzes=quizzes,
        attempt_counts=attempt_counts,
        last_stats=last_stats,
        in_progress_quiz_ids=in_progress,
    )


@app.route("/quizzes/<int:quiz_id>")
@login_required
def quizzes_detail(quiz_id: int):
    """Own quizzes or shared global quizzes; attempts are always for the current user."""
    sid = _get_system_quiz_user_id()
    qz = Quiz.query.options(selectinload(Quiz.questions)).filter_by(id=quiz_id).first()
    if not qz:
        flash("Quiz not found.")
        return redirect(url_for("quizzes_list"))
    allowed = qz.user_id == current_user.id or (
        qz.is_global and sid is not None and qz.user_id == sid
    )
    if not allowed:
        flash("Quiz not found.")
        return redirect(url_for("quizzes_list"))
    attempts = (
        QuizAttempt.query.filter_by(quiz_id=qz.id, user_id=current_user.id)
        .order_by(QuizAttempt.started_at.desc())
        .all()
    )
    n_questions = len(qz.questions)
    return render_template(
        "quiz_detail.html",
        active="quizzes",
        quiz=qz,
        attempts=attempts,
        n_questions=n_questions,
        is_global_quiz=bool(qz.is_global),
    )


@app.route("/quizzes/new", methods=["GET", "POST"])
@login_required
def quizzes_new():
    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        if not title:
            flash("Please enter a quiz title.")
            return redirect(url_for("quizzes_new"))
        slug = _slugify_quiz_name((request.form.get("slug") or "").strip() or title)
        if not slug:
            flash("Could not build a voice name from that title. Use letters or numbers.")
            return redirect(url_for("quizzes_new"))
        if Quiz.query.filter_by(user_id=current_user.id, slug=slug).first():
            flash("You already have a quiz with that voice name. Change the title or voice name.")
            return redirect(url_for("quizzes_new"))
        prompts = request.form.getlist("prompt")
        answers = request.form.getlist("answer")
        pairs = [(p.strip(), a.strip()) for p, a in zip(prompts, answers) if p.strip() and a.strip()]
        if not pairs:
            flash("Add at least one question and answer.")
            return redirect(url_for("quizzes_new"))
        qz = Quiz(
            user_id=current_user.id,
            title=title[:200],
            slug=slug,
            status="not_started",
        )
        db.session.add(qz)
        db.session.flush()
        _save_quiz_from_form(qz, title, slug, [x[0] for x in pairs], [x[1] for x in pairs])
        db.session.commit()
        flash("Quiz saved.")
        return redirect(url_for("quizzes_list"))
    return render_template("quiz_edit.html", active="quizzes", quiz=None, questions=[])


@app.route("/quizzes/<int:quiz_id>/edit", methods=["GET", "POST"])
@login_required
def quizzes_edit(quiz_id: int):
    qz = Quiz.query.filter_by(id=quiz_id, user_id=current_user.id).first()
    if not qz:
        flash("Quiz not found.")
        return redirect(url_for("quizzes_list"))
    if qz.is_global:
        flash("Built-in quizzes for everyone cannot be edited here.")
        return redirect(url_for("quizzes_list"))
    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        if not title:
            flash("Please enter a quiz title.")
            return redirect(url_for("quizzes_edit", quiz_id=quiz_id))
        slug = _slugify_quiz_name((request.form.get("slug") or "").strip() or title)
        if not slug:
            flash("Invalid voice name.")
            return redirect(url_for("quizzes_edit", quiz_id=quiz_id))
        other = Quiz.query.filter_by(user_id=current_user.id, slug=slug).first()
        if other and other.id != qz.id:
            flash("Another quiz already uses that voice name.")
            return redirect(url_for("quizzes_edit", quiz_id=quiz_id))
        prompts = request.form.getlist("prompt")
        answers = request.form.getlist("answer")
        pairs = [(p.strip(), a.strip()) for p, a in zip(prompts, answers) if p.strip() and a.strip()]
        if not pairs:
            flash("Add at least one question and answer.")
            return redirect(url_for("quizzes_edit", quiz_id=quiz_id))
        _save_quiz_from_form(qz, title, slug, [x[0] for x in pairs], [x[1] for x in pairs])
        db.session.commit()
        flash("Quiz updated.")
        return redirect(url_for("quizzes_list"))
    questions = (
        QuizQuestion.query.filter_by(quiz_id=qz.id)
        .order_by(QuizQuestion.sort_order, QuizQuestion.id)
        .all()
    )
    return render_template("quiz_edit.html", active="quizzes", quiz=qz, questions=questions)


@app.route("/quizzes/<int:quiz_id>/delete", methods=["POST"])
@login_required
def quizzes_delete(quiz_id: int):
    qz = Quiz.query.filter_by(id=quiz_id, user_id=current_user.id).first()
    if not qz:
        flash("Quiz not found.")
        return redirect(url_for("quizzes_list"))
    if qz.is_global:
        flash("Built-in quizzes cannot be deleted.")
        return redirect(url_for("quizzes_list"))
    db.session.delete(qz)
    db.session.commit()
    flash("Quiz deleted.")
    return redirect(url_for("quizzes_list"))


@app.route("/ask", methods=["POST"])
@login_required
def ask():
    """
    Voice UI: answers via the Alexa skill Lambda when ALEXA_LAMBDA_ARN is set, else optional Gemini.
    """
    app.logger.info("/ask POST from user=%s", getattr(current_user, "username", "?"))
    data = request.json
    question = data.get("question", "").strip()
    session_id = data.get("session_id")
    
    if not question:
        return jsonify({"answer": "Please ask a valid question."})
    
    if not session_id:
        return jsonify({"answer": "No active session. Please start Alexa first."}), 400

    conv_session = ConversationSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if not conv_session:
        return jsonify({"answer": "Session not found."}), 404

    quiz_reply = try_quiz_voice_reply(question, conv_session)
    if quiz_reply is not None:
        answer_text = quiz_reply
        _bk = "quiz_mode"
    else:
        answer_text = answer_for_voice_ui(question)
        _bk = voice_ask_backend()

    # Save to history (database)
    interaction = Interaction(session_id=conv_session.id, question=question, answer=answer_text)
    db.session.add(interaction)
    db.session.commit()

    return jsonify({
        "answer": answer_text,
        "session_id": conv_session.id,
        "ask_backend": _bk,
        "ask_backend_label": ask_backend_label(_bk),
    })

@app.route("/history/<int:session_id>", methods=["GET"])
@login_required
def get_history(session_id):
    """
    Returns all interactions for a given session.
    """
    conv_session = ConversationSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if not conv_session:
        return jsonify({"error": "Session not found"}), 404
    
    interactions = Interaction.query.filter_by(session_id=session_id).order_by(Interaction.timestamp).all()
    return jsonify([interaction.to_dict() for interaction in interactions])

@app.route("/sessions", methods=["POST"])
@login_required
def create_session():
    """
    Create a new conversation session.
    """
    # Create new session with current timestamp as title
    import datetime
    title = f"Session {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    conv_session = ConversationSession(
        user_id=current_user.id,
        title=title
    )
    db.session.add(conv_session)
    db.session.commit()
    
    return jsonify({
        'id': conv_session.id,
        'title': conv_session.title,
        'created_at': conv_session.created_at.isoformat()
    })

@app.route("/sessions", methods=["GET"])
@login_required
def get_sessions():
    """
    Get all conversation sessions for the current user.
    """
    sessions = ConversationSession.query.filter_by(user_id=current_user.id).order_by(ConversationSession.created_at.desc()).all()
    
    return jsonify([{
        'id': session.id,
        'title': session.title,
        'created_at': session.created_at.isoformat(),
        'interaction_count': len(session.interactions)
    } for session in sessions])

@app.route("/session/<int:session_id>", methods=["DELETE"])
@login_required
def delete_session(session_id):
    """
    Delete a session and all its interactions.
    """
    conv_session = ConversationSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if not conv_session:
        return jsonify({"error": "Session not found"}), 404
    
    db.session.delete(conv_session)
    db.session.commit()
    return jsonify({"success": True})


@app.route("/reports")
@login_required
def reports():
    """
    Returns learning reports for the current user.
    """
    sessions = ConversationSession.query.filter_by(user_id=current_user.id).all()
    activities = []
    for session in sessions:
        for interaction in session.interactions:
            activities.append({
                "timestamp": interaction.timestamp.isoformat(),
                "type": "QuestionAsked",
                "details": interaction.question,
                "session_title": session.title
            })
    return jsonify(activities)


@app.route('/export/<int:session_id>')
@login_required
def export_pdf(session_id):
    """
    Generate a simple PDF transcript of a session's conversation history.
    """
    conv_session = ConversationSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if not conv_session:
        return jsonify({"error": "Session not found"}), 404

    interactions = Interaction.query.filter_by(session_id=session_id).order_by(Interaction.timestamp).all()
    if not interactions:
        return jsonify({"error": "No records for session"}), 404

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'Conversation: {conv_session.title}', ln=True)
    pdf.ln(5)
    pdf.set_font('Arial', '', 12)

    for interaction in interactions:
        ts = interaction.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, f'[{ts}] Q: {interaction.question}', ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 8, f'A: {interaction.answer}')
        pdf.ln(2)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return send_file(buf, mimetype='application/pdf', download_name=f'{conv_session.title}_conversation.pdf')

@app.route('/interaction_model.json')
def interaction_model():
    """Return the bundled Alexa interaction model as a convenience."""
    return send_file('interaction_model.json', mimetype='application/json')


def _gtts_to_buffer(text: str) -> io.BytesIO:
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text, lang="en", slow=False).write_to_fp(buf)
    buf.seek(0)
    return buf


def _use_polly_tts() -> bool:
    return os.getenv("USE_POLLY_TTS", "").strip().lower() in ("1", "true", "yes")


def _polly_to_buffer(text: str) -> io.BytesIO:
    """Amazon Polly MP3. Not the proprietary Echo Alexa voice (unavailable outside Alexa products)."""
    import boto3

    region = (os.getenv("POLLY_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1").strip()
    # Default Joanna = en-US neural female; skills often use these Polly IDs in SSML <voice>.
    voice_id = os.getenv("POLLY_VOICE_ID", "Joanna").strip()
    engine = os.getenv("POLLY_ENGINE", "neural").strip()
    client = boto3.client("polly", region_name=region)
    resp = client.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId=voice_id,
        Engine=engine,
    )
    stream = resp["AudioStream"]
    data = stream.read()
    buf = io.BytesIO(data)
    buf.seek(0)
    return buf


def _tts_to_buffer(text: str) -> io.BytesIO:
    """Polly when USE_POLLY_TTS=1 (needs IAM polly:SynthesizeSpeech); else gTTS."""
    if _use_polly_tts():
        try:
            return _polly_to_buffer(text)
        except Exception as e:
            app.logger.warning("Polly TTS failed, using gTTS fallback: %s", e)
    return _gtts_to_buffer(text)


@app.route("/tts", methods=["GET", "POST"])
@login_required
def tts():
    """Return MP3 for browser playback: Amazon Polly (optional) or gTTS.

    Prefer POST JSON ``{"text": "..."}`` — long answers exceed safe GET URL length.
    GET ``?text=`` kept for simple links / backwards compatibility.
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
    else:
        text = (request.args.get("text") or "").strip()
    if not text:
        return jsonify({"error": "missing text"}), 400
    max_chars = int(os.getenv("TTS_MAX_CHARS", "4000"))
    text = text[:max_chars]
    # Polly synthesize limit per request is 3000 characters (neural)
    if _use_polly_tts():
        text = text[:3000]
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_tts_to_buffer, text)
            buf = fut.result(timeout=float(os.getenv("TTS_TIMEOUT_SEC", "120")))
    except FuturesTimeoutError:
        return jsonify({"error": "Text-to-speech timed out. Try a shorter answer."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return send_file(buf, mimetype="audio/mpeg")


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Starting Flask backend on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)