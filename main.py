from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
import time
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi import Query, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, validator
from datetime import datetime, timedelta
import uuid
import traceback
import io
import os
import shutil
import sqlite3
import sys
import tempfile
import re
import random
import base64
import json
import urllib.parse
from pydantic import BaseModel
import numpy as np
import httpx
from sentence_transformers import SentenceTransformer
from pypinyin import lazy_pinyin, Style
from gtts import gTTS
import eng_to_ipa as ipa
from ipa import ipa_to_word
import ollama
from openai import OpenAI
import main_chat
import requests
import aiohttp
from playwright.sync_api import sync_playwright
from jose import JWTError, jwt
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from docx import Document
import pdfplumber
import magic
import slp_training_tasks as slp_tasks
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from passlib.hash import bcrypt as passlib_bcrypt
import bcrypt as pybcrypt
from fastapi import Response
from fastapi.responses import FileResponse
from pathlib import Path
from LLM1_cn.AdditionPractice import detect_addition_error as detect_addition_error_1
from LLM1_cn.Distortion import detect_distortion_error
from LLM1_cn.InitialConsonantPractice import detect_initial_consonant_error
from LLM1_cn.Tone import detect_tone_error
from LLM1_cn.VowelFinal import detect_vowelfinal_error
from LLM1_cn.OmissionPractice import detect_omission_error as detect_omission_error_1
from LLM1_cn.SubstitutionPractice import detect_substitution_error as detect_substitution_error_1
from fastapi import FastAPI, HTTPException
from LLM_US.Substitution import detect_substitution_error
from LLM_US.Addition import detect_addition_error
from LLM_US.Omission import detect_omission_error
import logging
from Chinese.asr_llm_server import transcribe_bytes
import Chinese.asr_llm_server as asr_mod
from English_US.asr_llm_server import transcribe_bytes_en
from English_US.llm_call_gradio_client_local1 import analyze_record_payload
import uuid
from datetime import datetime
from English_US.asr_llm_server import transcribe_bytes_en as transcribe_bytes_en_full
import subprocess

uv_logger = logging.getLogger("uvicorn.error")
uv_logger.warning("[IMPORT DEBUG] Chinese.asr_llm_server file = %s", asr_mod.__file__)
uv_logger.warning("[IMPORT DEBUG] transcribe_bytes module = %s", transcribe_bytes.__module__)

from pydub import AudioSegment
from ws_audio import router as ws_audio_router

app = FastAPI()
app.include_router(ws_audio_router)

ESPEAK_DIR = r"eSpeak NG"
os.environ["PATH"] = ESPEAK_DIR + os.pathsep + os.environ.get("PATH", "")
BASE_DIR = Path(__file__).resolve().parent

USER_ROOT_DIR = "User" 
os.makedirs(USER_ROOT_DIR, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent / "English_US"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "Chinese"))

BASE_DIR = Path(__file__).resolve().parent
FFMPEG_BIN = (BASE_DIR / "ffmpeg" / "bin").resolve()

FFMPEG_EXE = str((FFMPEG_BIN / "ffmpeg.exe").resolve())
FFPROBE_EXE = str((FFMPEG_BIN / "ffprobe.exe").resolve())
LLM_MODEL_PATH = "./Chinese/model/Tiger-Gemma-9B-v3-Q3_K_M.gguf"

AudioSegment.converter = FFMPEG_EXE
AudioSegment.ffprobe = FFPROBE_EXE


os.environ["PATH"] = str(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")

print("[FFMPEG] bin =", FFMPEG_BIN)
print("[FFMPEG] ffmpeg exists?", os.path.exists(FFMPEG_EXE), FFMPEG_EXE)
print("[FFMPEG] ffprobe exists?", os.path.exists(FFPROBE_EXE), FFPROBE_EXE)
AudioSegment.ffprobe   = str(FFMPEG_BIN / "ffprobe.exe")

app = FastAPI(title="ASR+LLM Backend with DB")

# ✅ 掛載靜態路徑名稱未必要一致，但建議保持清楚
app.mount("/User", StaticFiles(directory=USER_ROOT_DIR), name="User")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

SECRET_KEY = "abc123"
ADMIN_USERS = ["admin", "bbb"]
ALGORITHM = "HS256"
ROUTE_DB = "route_knowledge.db"
FRONTEND_BASE = r"..\frontend\src"
FRONTEND_URL = "https://203.176.209.165:8443" 

ADMIN_LOCAL_ONLY = "admin" 
ONLY_ADMIN_UPLOAD = True
SLP_INACTIVE_CLOSE_MINUTES = 5
PRESENCE_TTL_SECONDS = 60  

TEST_WORDS_PER_SUBTYPE = 3 #this is cn
TEST_SLP_THRESHOLD = 1 #this is cn

TEST_EN_LANG = "en" #this is en
TEST_EN_WORDS_PER_SUBTYPE_DEFAULT = 3 #this is en
TEST_EN_SLP_THRESHOLD_DEFAULT = 2 #this is en
TEST_EN_SUPER_ORDER = ["Addition", "Omission", "Substitution"]
test_en_router = APIRouter(prefix="/api/test_en", tags=["test_en"])

SEND_USERNAME_COOKIE_DEFAULT = False
UPLOAD_AUDIO_DIR = "./uploads/audio"
UPLOAD_IMAGE_DIR = "./uploads/image"
UPLOAD_DOC_DIR = "uploaded_docs"
os.makedirs(UPLOAD_AUDIO_DIR, exist_ok=True)
os.makedirs(UPLOAD_IMAGE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DOC_DIR, exist_ok=True)

def levenshtein_distance(a, b) -> int:
    # a, b can be list (segments) or str
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,   # insertion
                prev[j - 1] + cost  # substitution
            )
        prev = cur
    return prev[m]

def normalize_difficulty(level: str) -> str:
    """
    DB records_analysis.difficulty 只接受:
      primary | secondary | advanced
    前端/ASR 可能送:
      小學 | 中學 | 大學 之類
    """
    s = (level or "").strip().lower()
    mapping = {
        "小學": "primary",
        "小学": "primary",
        "primary": "primary",

        "中學": "secondary",
        "中学": "secondary",
        "secondary": "secondary",

        "大學": "advanced",
        "大学": "advanced",
        "advanced": "advanced",
    }
    return mapping.get(s, "primary")

def save_records_analysis_v2(
    user_id: str,
    language: str,          # 例如 "Chinese"（不要用 "en"）
    test_type: str,         # super type，例如 "Substitution" / "Tone" / "Addition"...
    category: str,          # sub type，例如 pattern 或 bank key
    target_word: str,
    target_ipa: str,
    asr_ipa: str,
    asr_word: str,
    correct_rate,           # int 或 str 都可，我會轉 str
    check_results: dict,
    difficulty_level: str = "primary",
    score: int | None = None,
    severity: str | None = None,
):
    conn = get_db()
    cur = conn.cursor()
    rid = str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.execute("""
        INSERT INTO records_analysis
        (id, user_id, date,
         target_word, target_ipa, asr_ipa, asr_word,
         score, correct_rate, severity, difficulty,
         check_results, language, test_type, category, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        rid, user_id, now,
        target_word, target_ipa, asr_ipa, asr_word,
        score,
        str(correct_rate) if correct_rate is not None else None,
        severity,
        normalize_difficulty(difficulty_level),
        json.dumps(check_results, ensure_ascii=False),
        language,
        test_type,
        category,
        now
    ))

    conn.commit()
    conn.close()
    return rid

def climb_and_verify(url: str, keywords: list[str]) -> bool:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(ignore_https_errors=True)  # ✅ 忽略 SSL 問題
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")
        content = page.content()
        browser.close()
        return any(keyword.lower() in content.lower() for keyword in keywords)

router = APIRouter()

@app.on_event("startup")
def init_task_table_safe():
    try:
        ensure_task_table_v2()
        ensure_staff_verify_table()
        ensure_customer_service_tables()
        ensure_tests_table_v2_columns()
        migrate_practice_progress_unique_with_language()
        ensure_practice_progress_table()
        conn = get_db()
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(patients);")
        columns = [r[1] for r in cur.fetchall()]
        if "task_id" not in columns:
            cur.execute("ALTER TABLE patients ADD COLUMN task_id TEXT;")
            conn.commit()
            print("🧿 已自動為 patients 表新增欄位：task_id")
        conn.close()
    except Exception as e:
        print("⚠️ 初始化任務表失敗：", e)

sys.stdout.reconfigure(encoding='utf-8')

TTS_DIR = "tts_audio"
os.makedirs(TTS_DIR, exist_ok=True)

from fastapi.responses import StreamingResponse
import requests

# 測試首頁
@app.get("/")
async def root():
    return {"message": "Google TTS API is running. Try /google_tts/hello"}

@app.get("/routes")
async def get_routes():
    return [{"path": route.path, "name": route.name, "methods": list(route.methods)} for route in app.routes]
# 如果用 uvicorn run
# uvicorn this_file_name:app --host 0.0.0.0 --port 8001 --reload
    
class WordRequest(BaseModel):
    errorType: str
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def banks_json_to_error_map(data: dict) -> dict:
    """
    支援新格式：
      {
        "meta": {...},
        "banks": {
           "<bucket_name>": {"count":..., "items":[{"w":"..."}, ...]},
           ...
        }
      }
    轉成：
      {"<bucket_name>": ["word1","word2",...], ...}

    也支援舊格式：
      {"<bucket_name>": ["word1",...], ...}
    """
    if not isinstance(data, dict):
        return {}

    # NEW format
    if "banks" in data and isinstance(data.get("banks"), dict):
        out = {}
        for bucket, bank in data["banks"].items():
            if not isinstance(bank, dict):
                continue
            items = bank.get("items", [])
            if not isinstance(items, list):
                continue

            words = []
            seen = set()
            for it in items:
                if not isinstance(it, dict):
                    continue
                w = it.get("w")
                if isinstance(w, str):
                    ww = w.strip()
                    if ww and ww not in seen:
                        seen.add(ww)
                        words.append(ww)
            if words:
                out[str(bucket)] = words
        return out

    # OLD format
    out = {}
    for k, v in data.items():
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[str(k)] = v
    return out


def safe_load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not a JSON object at top-level")
    return obj

# 載入 Substitution.json
substitution_errors = banks_json_to_error_map(safe_load_json("LLM/Substitution.json"))
omission_errors_llm1 = banks_json_to_error_map(safe_load_json("LLM/Omission.json"))
addition_errors = banks_json_to_error_map(safe_load_json("LLM/Addition.json"))

print("DEBUG substitution_errors keys:", list(substitution_errors.keys())[:20])
print("DEBUG omission_errors_llm1 keys:", list(omission_errors_llm1.keys())[:20])
print("DEBUG addition_errors keys:", list(addition_errors.keys())[:20])

with open("LLM1/omission.json", "r", encoding="utf-8") as f:
    omission_errors = json.load(f)

print("DEBUG omission keys:", list(omission_errors.keys()))

with open("LLM1/substitution.json", "r", encoding="utf-8") as f:
    substitution_errors_llm1 = json.load(f)

with open("LLM1/Distortion.json", "r", encoding="utf-8") as f:
    distortion_errors_llm1 = json.load(f)

with open("LLM1/Addition.json", "r", encoding="utf-8") as f:
    Addition_errors_llm1 = json.load(f)
    ADDITION_BANK = Addition_errors_llm1

with open("LLM1/Tone.json", "r", encoding="utf-8") as f:
    Tone_errors_llm1 = json.load(f)

with open("LLM1/VowelFinal.json", "r", encoding="utf-8") as f:
    VowelFinal = json.load(f)

with open("LLM1/InitialConsonant.json", "r", encoding="utf-8") as f:
    InitialConsonant = json.load(f)
# ⚠️ 注意：這裡要取裡面的 "AdditionErrors"
slpErrors = {
    **substitution_errors,        # LLM/Substitution.json (banks -> map)
    **omission_errors,            # LLM1/omission.json (看起來是舊格式 dict)
    **substitution_errors_llm1,   # LLM1/substitution.json
    **addition_errors,            # ✅ 這裡直接合併整個 map，不要 ["AdditionErrors"]
    **omission_errors_llm1,       # LLM/Omission.json (banks -> map)
    **distortion_errors_llm1,
    **Addition_errors_llm1,
    **Tone_errors_llm1,
    **VowelFinal,
    **InitialConsonant,
}

def get_db():
    conn = sqlite3.connect("asr_llm.db")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

# ==============================
# ✅ Tests v2 (plan + progress) Migration & Helpers
# ==============================

def ensure_tests_table_v2_columns():
    # 你已經用 create_db_new.py 建好新 schema，這裡只保留 index（可選）
    conn = get_db()
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tests_patient_status_time ON tests(patient_id, status, updated_at);")
    conn.commit()
    conn.close()


def build_test_flat_queue(plan: list[dict]) -> list[dict]:
    """
    plan format: [{super_type, sub_type, words:[...]}]
    return flatQueue: [{index, super_type, sub_type, word}]
    """
    flat = []
    idx = 0
    for blk in plan or []:
        st = str(blk.get("super_type", "")).strip()
        sb = str(blk.get("sub_type", "")).strip()
        words = blk.get("words", []) or []
        for w in words:
            ww = str(w).strip()
            if not ww:
                continue
            flat.append({"index": idx, "super_type": st, "sub_type": sb, "word": ww})
            idx += 1
    return flat


def load_test_plan_obj(cur, test_id: str) -> dict:
    cur.execute("SELECT plan FROM tests WHERE id=?", (test_id,))
    row = cur.fetchone()
    if not row or not row[0]:
        return {}
    try:
        obj = json.loads(row[0])
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def compute_test_progress_cursor(cur, test_id: str) -> int:
    cur.execute("SELECT COUNT(DISTINCT question_index) FROM test_records WHERE test_id=?", (test_id,))
    return int(cur.fetchone()[0] or 0)

# ==============================
# ✅ Customer Service (DB + Helpers)
# ==============================

def compute_priority_by_age_minutes(age_minutes: int) -> int:
    # 0=low, 1=medium, 2=high
    if age_minutes <= 5:
        return 0
    if age_minutes <= 10:
        return 1
    return 2

def refresh_one_session_priority(cur, session_id: str):
    # 用 created_at 算等待分鐘數
    cur.execute("""
        SELECT priority,
               CAST((julianday('now') - julianday(created_at)) * 24 * 60 AS INTEGER) AS age_minutes
        FROM service_sessions
        WHERE id=?
    """, (session_id,))
    row = cur.fetchone()
    if not row:
        return

    old_p = int(row[0] or 0)
    age_minutes = int(row[1] or 0)
    new_p = compute_priority_by_age_minutes(age_minutes)

    # 只升不降（避免時間誤差造成下降）
    if new_p > old_p:
        cur.execute("UPDATE service_sessions SET priority=? WHERE id=?", (new_p, session_id))


def ensure_practice_progress_table():
    conn = get_db()
    cur = conn.cursor()

    # 先建立（兼容舊 DB 或新 DB）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS practice_progress (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        super_type TEXT NOT NULL,
        sub_type TEXT NOT NULL,
        chunk_size INTEGER NOT NULL DEFAULT 50,
        next_start_index INTEGER NOT NULL DEFAULT 0,
        last_range_end INTEGER NOT NULL DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES patients(id) ON DELETE CASCADE,
        UNIQUE(user_id, super_type, sub_type)
    )
    """)

    # ✅ Migration: 補 language 欄位（舊 DB 沒有就加）
    cur.execute("PRAGMA table_info(practice_progress);")
    cols = [r[1] for r in cur.fetchall()]
    if "language" not in cols:
        cur.execute("ALTER TABLE practice_progress ADD COLUMN language TEXT NOT NULL DEFAULT 'en';")
        conn.commit()
        print("🧿 practice_progress 已新增欄位：language (default='en')")

    conn.commit()
    conn.close()

def ensure_customer_service_tables():
    conn = get_db()
    cur = conn.cursor()

    # 1) service_sessions
    cur.execute("""
    CREATE TABLE IF NOT EXISTS service_sessions (
        id          TEXT PRIMARY KEY,
        patient_id  TEXT NOT NULL,
        slp_id      TEXT,

        topic       TEXT,
        priority    INTEGER NOT NULL DEFAULT 0,

        status      TEXT NOT NULL DEFAULT 'queued'
                    CHECK(status IN ('queued','assigned','open','closed','cancelled')),

        last_message_at      TIMESTAMP,
        last_slp_activity_at TIMESTAMP,

        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        closed_at   TIMESTAMP,
        closed_by   TEXT CHECK(closed_by IN ('patient','slp','system')),

        FOREIGN KEY(patient_id) REFERENCES patients(id) ON DELETE CASCADE,
        FOREIGN KEY(slp_id)     REFERENCES staff_users(id) ON DELETE RESTRICT
    );
    """)

    # 2) service_messages
    cur.execute("""
    CREATE TABLE IF NOT EXISTS service_messages (
        id           TEXT PRIMARY KEY,
        session_id   TEXT NOT NULL,
        sender_type  TEXT NOT NULL CHECK(sender_type IN ('patient','slp','system')),
        sender_id    TEXT,
        message      TEXT NOT NULL,
        message_type TEXT NOT NULL DEFAULT 'text'
                     CHECK(message_type IN ('text','system_event')),
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY(session_id) REFERENCES service_sessions(id) ON DELETE CASCADE
    );
    """)

    # 3) service_assignments
    cur.execute("""
    CREATE TABLE IF NOT EXISTS service_assignments (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        slp_id TEXT NOT NULL,
        assigned_by TEXT,
        assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        unassigned_at TIMESTAMP,
        reason TEXT,

        FOREIGN KEY(session_id) REFERENCES service_sessions(id) ON DELETE CASCADE,
        FOREIGN KEY(slp_id) REFERENCES staff_users(id) ON DELETE RESTRICT,
        FOREIGN KEY(assigned_by) REFERENCES staff_users(id) ON DELETE SET NULL
    );
    """)

    # 4) audit_logs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id TEXT PRIMARY KEY,
        actor_type TEXT NOT NULL CHECK(actor_type IN ('patient','slp','admin','system')),
        actor_id TEXT,
        action TEXT NOT NULL,
        resource_type TEXT NOT NULL,
        resource_id TEXT,
        reason TEXT,
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 5) slp_presence（關鍵：只有 is_waiting=1 才可被 auto allocate）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS slp_presence (
        slp_id TEXT PRIMARY KEY,
        is_online INTEGER NOT NULL DEFAULT 0 CHECK(is_online IN (0,1)),
        is_waiting INTEGER NOT NULL DEFAULT 0 CHECK(is_waiting IN (0,1)),
        last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        waiting_since TIMESTAMP,
        FOREIGN KEY(slp_id) REFERENCES staff_users(id) ON DELETE CASCADE
    );
    """)

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_service_sessions_status_created ON service_sessions(status, created_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_service_sessions_patient ON service_sessions(patient_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_service_sessions_slp ON service_sessions(slp_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_service_messages_session_time ON service_messages(session_id, created_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_service_assignments_session_time ON service_assignments(session_id, assigned_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_slp_presence_waiting ON slp_presence(is_waiting, last_seen_at);")

    conn.commit()
    conn.close()

def audit_log(actor_type, actor_id, action, resource_type, resource_id, reason=None, metadata=None):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO audit_logs
        (id, actor_type, actor_id, action, resource_type, resource_id, reason, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (str(uuid.uuid4()), actor_type, actor_id, action, resource_type, resource_id, reason, metadata))
    conn.commit()
    conn.close()

def slp_set_online_db(slp_id: str, online: bool):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO slp_presence (slp_id, is_online, is_waiting, last_seen_at, waiting_since)
        VALUES (?, ?, 0, CURRENT_TIMESTAMP, NULL)
        ON CONFLICT(slp_id) DO UPDATE SET
            is_online=excluded.is_online,
            last_seen_at=CURRENT_TIMESTAMP,
            is_waiting=CASE WHEN excluded.is_online=0 THEN 0 ELSE slp_presence.is_waiting END,
            waiting_since=CASE WHEN excluded.is_online=0 THEN NULL ELSE slp_presence.waiting_since END
    """, (slp_id, 1 if online else 0))
    conn.commit()
    conn.close()

def slp_wait_db(slp_id: str, waiting: bool):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO slp_presence (slp_id, is_online, is_waiting, last_seen_at, waiting_since)
        VALUES (?, 1, ?, CURRENT_TIMESTAMP, CASE WHEN ?=1 THEN CURRENT_TIMESTAMP ELSE NULL END)
        ON CONFLICT(slp_id) DO UPDATE SET
            is_online=1,
            is_waiting=?,
            last_seen_at=CURRENT_TIMESTAMP,
            waiting_since=CASE WHEN ?=1 THEN COALESCE(slp_presence.waiting_since, CURRENT_TIMESTAMP) ELSE NULL END
    """, (slp_id, 1 if waiting else 0, 1 if waiting else 0, 1 if waiting else 0, 1 if waiting else 0))
    conn.commit()
    conn.close()

def slp_heartbeat_db(slp_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE slp_presence SET last_seen_at=CURRENT_TIMESTAMP WHERE slp_id=?", (slp_id,))
    conn.commit()
    conn.close()

def pick_waiting_slp_id(cur) -> Optional[str]:
    cur.execute("""
        SELECT slp_id
        FROM slp_presence
        WHERE is_online=1 AND is_waiting=1
          AND last_seen_at >= datetime('now', ?)
        ORDER BY COALESCE(waiting_since, last_seen_at) ASC
        LIMIT 1
    """, (f"-{PRESENCE_TTL_SECONDS} seconds",))
    row = cur.fetchone()
    return row[0] if row else None

def patient_inactive_after_slp_reply(cur, session_id: str, minutes: int) -> bool:
    cur.execute("""
        SELECT
          MAX(CASE WHEN sender_type='patient' THEN created_at END) AS last_patient_at,
          MAX(CASE WHEN sender_type='slp' THEN created_at END)     AS last_slp_at
        FROM service_messages
        WHERE session_id=?
    """, (session_id,))
    row = cur.fetchone()
    last_patient_at, last_slp_at = row[0], row[1]

    # SLP 沒講過話，不算病人沒回覆
    if not last_slp_at:
        return False

    # 病人最後訊息時間 >= SLP 最後訊息時間，代表病人已回覆（或比 SLP 新），不允許用「病人逾時」關單
    if last_patient_at and str(last_patient_at) >= str(last_slp_at):
        return False

    # 以病人最後訊息為基準算逾時
    # 若病人沒有訊息（理論上你有 first_message，所以很少發生），則用 SLP 最後訊息算
    base_time = last_patient_at or last_slp_at

    cur.execute("""
        SELECT CAST((julianday('now') - julianday(?)) * 24 * 60 AS INTEGER)
    """, (base_time,))
    mins = int(cur.fetchone()[0] or 0)

    return mins >= int(minutes)

def system_auto_allocate_one_db() -> Optional[dict]:
    """
    系統派 1 張 queued 給 1 位等待中的 SLP
    - 若沒工單或沒 waiting SLP => None
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        # 找最舊 queued
        cur.execute("""
            SELECT id
            FROM service_sessions
            WHERE status='queued'
            ORDER BY created_at ASC
            LIMIT 1
        """)
        sess = cur.fetchone()
        if not sess:
            return None

        slp_id = pick_waiting_slp_id(cur)
        if not slp_id:
            return None

        session_id = sess[0]

        # 原子更新：避免重複派單
        cur.execute("""
            UPDATE service_sessions
            SET slp_id=?, status='assigned'
            WHERE id=? AND status='queued'
        """, (slp_id, session_id))
        if cur.rowcount != 1:
            return None

        # 寫入 assignment 記錄
        cur.execute("""
            INSERT INTO service_assignments (id, session_id, slp_id, assigned_by, reason)
            VALUES (?, ?, ?, NULL, 'AUTO_ALLOCATE')
        """, (str(uuid.uuid4()), session_id, slp_id))

        # 一次接一單：派到就關掉 waiting（你若要多單，把這段移除）
        cur.execute("""
            UPDATE slp_presence
            SET is_waiting=0, waiting_since=NULL, last_seen_at=CURRENT_TIMESTAMP
            WHERE slp_id=?
        """, (slp_id,))

        conn.commit()

        audit_log("system", None, "AUTO_ALLOCATE", "service_session", session_id,
                  metadata=json.dumps({"slp_id": slp_id}, ensure_ascii=False))

        return {"session_id": session_id, "slp_id": slp_id}
    finally:
        conn.close()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def pinyin_to_chinese_diff(asr_pinyin: str, target_pinyin: str, target_word: str) -> str:
    """
    把 ASR 拼音和 Target IPA 比對：
    - 一樣的地方直接用 target_word 對應字
    - 不一樣的地方，用拼音轉換成近似字（取第一個候選字）
    """
    asr_list = asr_pinyin.split()
    tgt_list = target_pinyin.split()
    result_chars = []

    for i, tgt in enumerate(tgt_list):
        if i < len(asr_list):
            asr = asr_list[i]
            if asr == tgt:
                # ✅ 取 target_word 對應位置的字
                if i < len(target_word):
                    result_chars.append(target_word[i])
                else:
                    result_chars.append("?")
            else:
                # ❌ 錯誤音節 → 嘗試用拼音轉漢字
                # 去掉數字聲調 (ma1 -> ma)
                base_pinyin = ''.join([c for c in asr if not c.isdigit()])
                candidates = lazy_pinyin(base_pinyin, style=Style.NORMAL)
                result_chars.append(candidates[0] if candidates else "?")
        else:
            # 用戶少念了 → 空
            result_chars.append("_")

    return "".join(result_chars)

def load_react_files():
    paths = []
    app_path = os.path.join(FRONTEND_BASE, "App.js")
    if os.path.exists(app_path):
        with open(app_path, "r", encoding="utf-8") as f:
            paths.append(("App.js", f.read()))
    pages_dir = os.path.join(FRONTEND_BASE, "pages")
    for root, dirs, files in os.walk(pages_dir):
        for file in files:
            if file.endswith((".js", ".jsx")):
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    paths.append((file, f.read()))
    return paths

def analyze_with_llm(files):
    prompt = "這是 React 專案的程式碼，請分析 routes (路徑) 與對應頁面，例如 `/profile → Profile`。\n\n"
    for filename, content in files:
        prompt += f"\n=== {filename} ===\n{content}\n"
    resp = ollama.chat(
        model="gpt-oss-20b",
        messages=[{"role":"user","content":prompt}],
        options={"num_predict": 800}
    )
    return resp["message"]["content"]

def init_path_knowledge(reset=True):
    files = load_react_files()
    structure = analyze_with_llm(files)
    conn = sqlite3.connect(ROUTE_DB)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS knowledge (id TEXT PRIMARY KEY, content TEXT)")
    if reset:
        cur.execute("DELETE FROM knowledge")
    cur.execute("INSERT INTO knowledge (id, content) VALUES (?, ?)", (str(uuid.uuid4()), structure))
    conn.commit()
    conn.close()
    print("✅ 路徑結構知識已重新建立")

def detect_language(text: str):
    chinese_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    english_chars = sum(1 for ch in text if ch.isalpha())
    if chinese_chars > english_chars:
        return "zh"
    return "en"

def ask_about_route(user_question: str):
    # STEP1 讀 DB 知識
    conn = sqlite3.connect(ROUTE_DB)
    cur = conn.cursor()
    cur.execute("SELECT content FROM knowledge LIMIT 1")
    row = cur.fetchone()
    conn.close()

    if not row:
        return "⚠️ 還沒有路徑知識，請先啟動伺服器建立"

    structure = row[0]
    lang = detect_language(user_question)

    # 系統指令
    role_instruction = (
        "你是一個網站導航助理，你需要幫助使用者操作這個網站：https://203.176.209.165:8443/ 。請用簡單中文分步驟告訴用戶要怎麼點擊才能到達目標頁面。"
        if lang == "zh" else 
        "You are a website navigation assistant. Your task is to guide users on this website: https://203.176.209.165:8443/ . Please reply in simple English with clear step‑by‑step navigation instructions."
    )

    # STEP2：LLM 找候選 path
    prompt = f"""
已知 React 專案的路徑結構：

{structure}

使用者問題：{user_question}

請只回覆最可能的 path (例如 `/settings/password`)，不要加多餘解釋。
"""
    resp = ollama.chat(
        model="gpt-oss-20b",
        messages=[
            {"role": "system", "content": role_instruction},
            {"role": "user", "content": prompt},
        ],
        options={"num_predict": 200},
    )
    candidate_path = resp["message"]["content"].strip()

    # STEP3：包裝成「操作指引」
    if candidate_path.startswith("/"):
        full_url = f"{FRONTEND_URL}{candidate_path}"

        # 簡單驗證頁面是否存在
        if climb_and_verify(full_url, ["密碼", "password", "修改", "設定", "change"]):
            if lang == "zh":
                return f"✅ 請打開網站選單，找到【{candidate_path}】頁面，點擊即可進入並操作。"
            else:
                return f"✅ Please open the website menu, find **{candidate_path}**, and click it to access the page."
        else:
            if lang == "zh":
                return f"⚠️ 找到了 {candidate_path}，但檢查後似乎沒有相關功能。"
            else:
                return f"⚠️ Found {candidate_path}, but no related function was detected."

    # STEP4：fallback 常見頁面
    fallback_paths = ["/settings", "/account", "/profile", "/user", "/config"]
    for path in fallback_paths:
        full_url = f"{FRONTEND_URL}{path}"
        if climb_and_verify(full_url, ["密碼", "password", "修改", "設定", "change"]):
            if lang == "zh":
                return f"✅ 雖然沒有找到精準頁面，但請嘗試點擊網站選單中的【{path}】，應該能進行相關操作。"
            else:
                return f"✅ The system did not return an exact page, but please try clicking **{path}** in the menu to perform the action."

    # STEP5：完全沒找到
    return "❌ 系統中沒有找到符合的頁面" if lang == "zh" else "❌ No matching page found in the system."

def get_current_username_from_db(token: str = Depends(oauth2_scheme)):
    """
    透過 token.sub (user_id) 從 patients 資料表查 username
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT username FROM patients WHERE id=?", (user_id,))
        row = cur.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=401, detail="User not found in DB")

        return row[0]  # 真正的 username
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")

# ➋ 封裝驗證函數
def get_current_user_id(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")

class UpdateMeRequest(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    diagnosis: Optional[str] = None
    email: Optional[str] = None
    language: Optional[str] = None

    old_password: Optional[str] = None
    new_password: Optional[str] = None

    @validator("age", pre=True)
    def parse_age(cls, v):
        # 容錯：允許前端送 "20" / "" / None
        if v is None:
            return None
        if isinstance(v, str):
            vv = v.strip()
            if vv == "":
                return None
            if vv.isdigit():
                return int(vv)
        return v


@app.post("/api/patient/update-me")
def update_me(
    data: UpdateMeRequest,
    current_user_id: str = Depends(get_current_user_id),
):
    """
    ✅ 只用 token(sub=user_id) 更新自己資料，不需要 username
    - 改密碼才需要 old_password + new_password
    - 改 email 時自動把 verify 設為 0（重新驗證）
    """
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT password_hash FROM patients WHERE id=?", (current_user_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    stored_hash = row[0]

    updates = []
    params = []

    if data.name is not None:
        updates.append("name=?")
        params.append(data.name)

    if data.age is not None:
        updates.append("age=?")
        params.append(data.age)

    if data.diagnosis is not None:
        updates.append("diagnosis=?")
        params.append(data.diagnosis)

    if data.language is not None:
        updates.append("language=?")
        params.append(data.language)

    if data.email is not None:
        updates.append("email=?")
        params.append(data.email)

        # email 改了 => 重新驗證
        updates.append("verify=?")
        params.append(0)

    # 改密碼（只有這個需要驗 old_password）
    if data.new_password:
        if not data.old_password or not verify_password(data.old_password, stored_hash):
            conn.close()
            raise HTTPException(status_code=400, detail="Old password incorrect")
        updates.append("password_hash=?")
        params.append(hash_password(data.new_password))

    if not updates:
        conn.close()
        return {"msg": "No fields updated"}

    sql = f"UPDATE patients SET {', '.join(updates)} WHERE id=?"
    params.append(current_user_id)
    cur.execute(sql, tuple(params))
    conn.commit()

    cur.execute("""
        SELECT id, username, name, age, diagnosis, email, language, integral, verify, user_picture
        FROM patients WHERE id=?
    """, (current_user_id,))
    user_row = cur.fetchone()
    conn.close()

    return {
        "msg": "Profile updated successfully",
        "user": {
            "id": user_row[0],
            "username": user_row[1],
            "name": user_row[2],
            "age": user_row[3],
            "diagnosis": user_row[4],
            "email": user_row[5],
            "language": user_row[6],
            "integral": user_row[7],
            "verify": user_row[8],
            "user_picture": user_row[9],
        }
    }

def hash_password(password: str) -> str:
    pw = password.encode("utf-8")
    salt = pybcrypt.gensalt(rounds=12)
    return pybcrypt.hashpw(pw, salt).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    return pybcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

from English_US.asr_llm_server import (
    load_and_denoise,
    recognize_word,
    normalize_ipa,
    pronunciation_dict,
)
from English_US.asr_llm_server import word_to_ipa_or_star

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class DBConnections:
    def __init__(self):
        self.db1 = sqlite3.connect("asr_llm.db")
        self.db2 = sqlite3.connect("asr_llm.db")

    def close(self):
        self.db1.close()
        self.db2.close()

@app.get("/api/checkin/history")
def get_checkin_history(current_user_id: str = Depends(get_current_user_id)):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT date FROM checkins WHERE user_id=? ORDER BY date", (current_user_id,))
    rows = cur.fetchall()
    history = [r[0] for r in rows]

    today = datetime.now().strftime("%Y-%m-%d")
    todayCheckedIn = today in history

    # 🔥 計算連續天數
    streak = 0
    for d in reversed(history):
        if d == (datetime.now().date() - timedelta(days=streak)).strftime("%Y-%m-%d"):
            streak += 1
        else:
            break
    print("DEBUG history for user:", current_user_id, history)        
    conn.close()
    return {"history": history, "streak": streak, "todayCheckedIn": todayCheckedIn}

# -------------------------------
# 📌 簽到
# -------------------------------
@app.post("/api/checkin")
def checkin(current_user_id: str = Depends(get_current_user_id)):
    conn = get_db()
    cur = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")

    # 先檢查今天是否已簽到
    cur.execute("SELECT 1 FROM checkins WHERE user_id=? AND date=?", (current_user_id, today))
    already = cur.fetchone() is not None

    if not already:
        # 1) 寫入簽到紀錄
        cur.execute(
            "INSERT INTO checkins (id, user_id, date) VALUES (?, ?, ?)",
            (str(uuid.uuid4()), current_user_id, today)
        )

        # 2) ✅ 加 10 分（只在首次簽到才加）
        cur.execute(
            "UPDATE patients SET integral = COALESCE(integral, 0) + 10 WHERE id=?",
            (current_user_id,)
        )

        conn.commit()

    # 查返最新紀錄
    cur.execute("SELECT date FROM checkins WHERE user_id=? ORDER BY date", (current_user_id,))
    rows = cur.fetchall()
    history = [r[0] for r in rows]

    todayCheckedIn = today in history

    # 計算連續天數
    streak = 0
    for d in reversed(history):
        if d == (datetime.now().date() - timedelta(days=streak)).strftime("%Y-%m-%d"):
            streak += 1
        else:
            break

    conn.close()
    return {
        "history": history,
        "streak": streak,
        "todayCheckedIn": todayCheckedIn,
        "added_integral": 10 if not already else 0   # ✅ 可選：回傳這次加了幾分
    }

@app.get("/db-test")
def db_test():
    dbs = DBConnections()
    cur1 = dbs.db1.cursor()
    cur2 = dbs.db2.cursor()

    cur1.execute("SELECT COUNT(*) FROM patients")
    cur2.execute("SELECT COUNT(*) FROM patients")

    r1 = cur1.fetchone()[0]
    r2 = cur2.fetchone()[0]

    dbs.close()
    return {"asr_llm.db patients": r1, "asr_llm.db patients": r2}

@app.delete("/auth/delete")
def delete_account(username: str = Query(...)):
    conn = get_db()
    cur = conn.cursor()

    # 檢查使用者是否存在
    cur.execute("SELECT id FROM patients WHERE username=?", (username,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    patient_id = row[0]

    # 刪掉 attempts → sessions → patient
    cur.execute("DELETE FROM attempts WHERE session_id IN (SELECT id FROM sessions WHERE patient_id=?)", (patient_id,))
    cur.execute("DELETE FROM sessions WHERE patient_id=?", (patient_id,))
    cur.execute("DELETE FROM patients WHERE id=?", (patient_id,))

    conn.commit()
    conn.close()

    return {"msg": f"Account {username} deleted successfully"}
    

from sqlite3 import Row  # 可選

@app.get("/auth/me_llm1")
def get_user_llm1(username: str = Query(..., alias="username")):
    token = username
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")

    conn = get_db()
    # ✅ 令 fetch 到嘅 row 可以用欄名轉 dict
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM records_analysis
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()

    # ✅ 將每筆 Row 轉成 dict（包含 records_analysis 全部欄位）
    records = [dict(r) for r in rows]

    return {"records": records}

def calc_correct_rate(asr: str, target: str) -> int:
    a = [x for x in (asr or "").strip().split() if x]
    b = [x for x in (target or "").strip().split() if x]
    if not a or not b:
        return 0
    dist = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    return round(((max_len - dist) / max_len) * 100) if max_len > 0 else 0
#---------TTS web--------#

@app.get("/google_tts/{word}")
async def google_tts(word: str):
    """
    Google TTS for English words (path parameter)
    """
    try:
        tts_url = "https://translate.google.com/translate_tts"
        lang = "en"

        encoded_word = urllib.parse.quote(word, safe="")
        full_url = f"{tts_url}?ie=UTF-8&q={encoded_word}&tl={lang}&client=gtx"

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
            "Referer": "https://translate.google.com/",
            "Range": "bytes=0-",
        }

        r = requests.get(full_url, headers=headers, stream=True, allow_redirects=True)

        if r.status_code not in (200, 206):
            raise HTTPException(status_code=500, detail=f"Google TTS failed: {r.status_code}")

        if "audio" not in r.headers.get("Content-Type", ""):
            raise HTTPException(status_code=500, detail="Google TTS did not return audio")

        return StreamingResponse(
            r.iter_content(chunk_size=1024),
            media_type="audio/mpeg",
            headers={"Content-Disposition": f'inline; filename="{word}.mp3"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")
    

@app.get("/google_tts_ch")
async def google_tts_ch(word: str = Query(..., description="必須是中文詞語")):
    """
    使用 Google Translate TTS endpoint 代理，直接回傳 MP3
    """
    url = "https://translate.google.com/translate_tts"
    params = {
        "ie": "UTF-8",
        "tl": "zh-CN",   # 中文
        "client": "tw-ob",
        "q": word,
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, params=params, headers=headers)
    if r.status_code != 200:
        return {"status": r.status_code, "error": "Google TTS request failed"}

    return StreamingResponse(io.BytesIO(r.content),
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'inline; filename="audio.mp3"'}
    )

#---------TTS web--------#

#------Cet word from LLM1 Chinese -----#
@app.get("/get_word_from_tone")
async def get_word_from_tone(errorType: str = Query(..., description="Tone error type")):
    """
    從 Tone.json 根據 errorType 抽一個詞回傳
    """
    with open("LLM1/Tone.json", "r", encoding="utf-8") as f:
        tone_errors = json.load(f)

    if not errorType or errorType not in tone_errors:
        raise HTTPException(status_code=400, detail="Invalid errorType")

    word = random.choice(tone_errors[errorType])
    return {"word": word}

@app.get("/get_word_from_addition")
async def get_word_from_addition(errorType: str = Query(..., description="Addition error type")):
    """
    從 Addition.json 根據 errorType 抽一個詞回傳
    - errorType: "声母添加" / "韵母添加" / "韵尾添加" / "插音"
    """
    if not errorType or errorType not in Addition_errors_llm1:
        raise HTTPException(status_code=400, detail="Invalid errorType")

    word = random.choice(Addition_errors_llm1[errorType])
    return {"word": word}


@app.get("/get_word_from_distortion")
async def get_word_from_distortion(errorType: str = Query(..., description="Distortion error type")):
    """
    從 Distortion.json 根據 errorType 抽一個詞回傳
    """
    if not errorType or errorType not in distortion_errors_llm1:
        raise HTTPException(status_code=400, detail="Invalid errorType")

    word = random.choice(distortion_errors_llm1[errorType])
    return {"word": word}

@app.get("/get_word_from_omission")
async def get_word_from_omission(errorType: str = Query(..., description="Omission error type")):
    if not errorType or errorType not in omission_errors:
        raise HTTPException(status_code=400, detail="Invalid errorType")

    word = random.choice(omission_errors[errorType])
    return {"word": word}


class VerifyCodeRequest(BaseModel):
    username: str
    code: str

@app.post("/auth/verify-code")
def verify_user_code(data: VerifyCodeRequest):
    """
    支援使用者輸入 username 或 email 驗證帳號，
    比對驗證碼後設定 verify=1。
    """
    conn = get_db()
    cur = conn.cursor()

    # 1️⃣ 嘗試找到匹配的驗證碼紀錄
    cur.execute("""
        SELECT code, username
        FROM email_verifies
        WHERE username = ?
           OR username IN (SELECT username FROM patients WHERE email = ?)
    """, (data.username, data.username))
    row = cur.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Verification code not found")

    db_code, username = row

    if str(data.code).strip() != str(db_code).strip():
        conn.close()
        raise HTTPException(status_code=400, detail="Incorrect verification code")

    # 2️⃣ 更新，若 username 對不到，用 email 也嘗試一次
    cur.execute("""
        UPDATE patients
        SET verify = 1
        WHERE username = ? OR email = ?
    """, (username, data.username))
    conn.commit()

    # 3️⃣ 驗證確實寫入
    cur.execute("SELECT verify FROM patients WHERE username = ? OR email = ?", (username, data.username))
    v = cur.fetchone()

    if not v or int(v[0]) != 1:
        conn.close()
        raise HTTPException(status_code=500, detail="⚠️ verify 欄位未更新，請確認帳號匹配")

    # 4️⃣ 成功後刪除驗證碼
    cur.execute("DELETE FROM email_verifies WHERE username = ?", (username,))
    conn.commit()
    conn.close()

    return {"msg": "✅ 驗證成功，帳號已啟用！"}

# -----------------------------
# API: 取得一個韻母錯誤詞
# -----------------------------
@app.get("/get_word_from_vowel_final")
async def get_word_from_vowel_final(errorType: str):
    """
    從 VowelFinal.json 隨機抽取一個詞
    errorType 必須是 VowelFinal.json 裡的 key (單元音替代 / 複合韻母錯誤)
    """
    try:
        if errorType not in VowelFinal:
            return JSONResponse(
                status_code=400,
                content={"detail": f"未知的韻母錯誤類型: {errorType}"}
            )

        word = random.choice(VowelFinal[errorType])
        return {"word": word}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"取詞失敗: {str(e)}"}
        )

#------Cet word from LLM1 -----#

#------Chinese-----#

def save_analysis_record(
    user_id: str,
    language: str,
    target_word: str,
    target_ipa: str,
    asr_ipa: str,
    asr_word: str,
    correct_rate: int,
    difficulty: str,
    check_results: dict,
    test_type: str,
    category: Optional[str] = None,
    score: Optional[int] = None,
):
    conn = get_db()
    cur = conn.cursor()
    now = datetime.now()
    record_id = str(uuid.uuid4())

    cur.execute("""
        INSERT INTO records_analysis (
            id, user_id, date,
            target_word, target_ipa, asr_ipa, asr_word,
            score, correct_rate, difficulty, check_results,
            language, test_type, category, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record_id,
        user_id,
        now.strftime("%Y-%m-%d %H:%M:%S"),
        target_word,
        target_ipa,
        asr_ipa,
        asr_word,
        score,
        correct_rate,
        difficulty,
        json.dumps(check_results, ensure_ascii=False),
        language,
        test_type,
        category,
        now.strftime("%Y-%m-%d %H:%M:%S"),
    ))

    conn.commit()
    conn.close()
    
def to_py_bool(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            # 0-d or 1 element -> item; multi-element -> any
            return bool(x.item()) if x.numel() == 1 else bool(x.any().item())
    except Exception:
        pass

    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return bool(x.item()) if x.size == 1 else bool(x.any())
    except Exception:
        pass

    return bool(x)

def pinyin_syllable_list(s: str) -> list[str]:
    # reuse your normalize_pinyin_list if you like
    s = (s or "").strip().lower().replace("/", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.split() if s else []

def normalize_bool(x):
    return bool(x)

SUPER_TYPE_TO_DETECTORS = {
    # 你前端 category 傳入嘅 super type
    "Substitution": ["substitution"],
    "Omission": ["omission"],
    "Addition": ["addition"],
    "Tone": ["tone"],
    "Distortion": ["distortion"],
    "InitialConsonant": ["initial"],
    "VowelFinal": ["vowel_final"],
    "Pinyin": ["tone"],  # 例：你 PinyinPractice 可能想 tone 做主
}

def severity_from_subtype_num(n: int | None):
    if not n or n <= 0:
        return None
    if n == 1:
        return "slight"
    if n in (2, 3):
        return "medium"
    return "high"   # >=4


@app.post("/analyze_substitution")
async def analyze_substitution(
    file: UploadFile = File(...),
    target_word: str = Form(...),
    username: str = Form("guest"),
    subtype: int = Form(0),  # legacy; ignored
    category: str = Form("Substitution"),
    pattern: str = Form("Substitution_initial_consonant"),
    checks: str = Form(""),  # legacy; ignored (substitution-only)
    current_user_id: str = Depends(get_current_user_id),
    difficulty_level: str = Form("小學"),
):
    """
    ✅ Substitution-only (script) version (aligned with /analyze_addition)

    - Only run Substitution detector (rule-based; no LLM).
      final_label: no_problem | slp | 亂答
      type (subtype): 声母替代 | 韵母替代 | 声调替代 (only if slp)
      severity: slight | middle | high              (only if slp)

    - Still compute:
        correct_rate (0~100)
        score (0~10) = round(correct_rate / 10)

    - status:
        correct if IPA exactly matches; else wrong
        (same as addition endpoint)
    """
    try:
        # -----------------------------
        # 1) read audio
        # -----------------------------
        content = await file.read()

        # -----------------------------
        # 2) ASR
        # -----------------------------
        asr_result = transcribe_bytes(
            audio_bytes=content,
            filename=file.filename,
            target_word=target_word,
            difficulty_level=difficulty_level,
        )

        asr_ipa = asr_result.get("asr_ipa", "") or ""
        target_ipa = (asr_result.get("target_ipa", "") or "").replace("/", "") or ""

        asr_word = asr_result.get("asr_word", "") or ""
        if not asr_word:
            if asr_ipa.strip() == target_ipa.strip():
                asr_word = target_word
            else:
                asr_word = pinyin_to_chinese_diff(asr_ipa, target_ipa, target_word)

        # -----------------------------
        # 3) accuracy + score points
        # -----------------------------
        correct_rate = calc_correct_rate(asr_ipa, target_ipa)   # 0~100
        score_points = int(round(correct_rate / 10))            # 0~10

        # -----------------------------
        # 4) meta off_target
        # -----------------------------
        off = compute_off_target(target_ipa, asr_ipa)
        is_off_target = bool(off.get("is_off_target"))

        # -----------------------------
        # 5) Substitution detector only (script-based)
        # -----------------------------
        sub_res = detect_substitution_error_1(
            target_word,
            asr_ipa,
            llm=None,                 # ✅ disable LLM explicitly
            off_target=off,           # ✅ let detector gate
            correct_rate=correct_rate,
        )

        check_results = {
            "meta": {
                "off_target": off,
                "enabled_checks": ["substitution"],
                "pattern": pattern,
                "category": category,
                "llm_model_repo": None,
                "llm_model_file": None,
            },
            "substitution": sub_res,
        }

        # -----------------------------
        # 6) status (substitution-only)
        # -----------------------------
        ipa_match = asr_ipa.strip() == target_ipa.strip()
        status = "correct" if ipa_match else "wrong"

        # -----------------------------
        # 7) severity (only when slp)
        # -----------------------------
        sub_final = str((sub_res or {}).get("final_label") or "").strip().lower()
        severity_value = (sub_res or {}).get("severity") if sub_final == "slp" else None
        if severity_value not in ("slight", "middle", "high"):
            severity_value = None

        # -----------------------------
        # 8) DB save
        # -----------------------------
        record_id = save_records_analysis_v2(
            user_id=current_user_id,
            language="Chinese",
            test_type=category,
            category=pattern,
            target_word=target_word,
            target_ipa=target_ipa,
            asr_ipa=asr_ipa,
            asr_word=asr_word,
            correct_rate=correct_rate,
            check_results=check_results,
            difficulty_level=difficulty_level,
            score=score_points,
            severity=severity_value,
        )

        # -----------------------------
        # 9) add points
        # -----------------------------
        if score_points > 0:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "UPDATE patients SET integral = COALESCE(integral,0) + ? WHERE id=?",
                (score_points, current_user_id),
            )
            conn.commit()
            conn.close()

        # -----------------------------
        # 10) return
        # -----------------------------
        return JSONResponse({
            "target_word": target_word,
            "asr_ipa": asr_ipa,
            "revise_ipa": target_ipa,
            "asr_word": asr_word,
            "correct_rate": correct_rate,
            "status": status,
            "check_results": check_results,
            "difficulty_level": difficulty_level,
            "severity": severity_value,
            "is_off_target": is_off_target,
            "score": score_points,
            "record_id": record_id,
            "patient_explain": (
                (check_results.get("substitution") or {}).get("patient_explain")
                or patient_friendly_explain(check_results)
            ),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Substitution analysis failed: {str(e)}")

# -----------------------------
# API: 分析韻母錯誤
# -----------------------------
@app.post("/analyze_vowel_final")
async def analyze_vowelfinal(
    file: UploadFile = File(...),
    target_word: str = Form(...),
    subtype: int = Form(0),          # legacy; ignored (match Addition behavior)
    username: str = Form("guest"),
    category: str = Form("VowelFinal"),
    pattern: str = Form("VowelFinal_error"),
    checks: str = Form(""),          # legacy; ignored (vowel-final-only)
    current_user_id: str = Depends(get_current_user_id),
    difficulty_level: str = Form("小學"),
):
    """
    ✅ VowelFinal-only (script) version — aligned to /analyze_addition & /analyze_tone & /analyze_omission

    - Only run VowelFinal detector (rule-based; no LLM).
      final_label: no_problem | slp | 亂答
      type (subtype): 单元音替代 | 复合韵母错误 | 韵母替代(可选fallback) (only if slp)
      severity: slight | middle | high                               (only if slp)

    - Still compute:
        correct_rate (accuracy, 0~100)
        score (0~10) = round(correct_rate / 10)

    - status:
        correct if asr_ipa == target_ipa
        wrong   otherwise
.

    - severity:
        only from vowel_final.severity AND only when final_label == slp
    """
    try:
        # -----------------------------
        # 1) read audio
        # -----------------------------
        content = await file.read()

        # -----------------------------
        # 2) ASR
        # -----------------------------
        asr_result = transcribe_bytes(
            audio_bytes=content,
            filename=file.filename,
            target_word=target_word,
            difficulty_level=difficulty_level,
        )

        # -----------------------------
        # 3) results
        # -----------------------------
        asr_ipa = asr_result.get("asr_ipa", "") or ""
        target_ipa = (asr_result.get("target_ipa", "") or "").replace("/", "") or ""

        asr_word = asr_result.get("asr_word", "") or ""
        if not asr_word:
            if asr_ipa.strip() == target_ipa.strip():
                asr_word = target_word
            else:
                asr_word = pinyin_to_chinese_diff(asr_ipa, target_ipa, target_word)

        # -----------------------------
        # 4) accuracy + score points
        # -----------------------------
        correct_rate = calc_correct_rate(asr_ipa, target_ipa)  # 0~100
        score_points = int(round(correct_rate / 10))           # 0~10

        # -----------------------------
        # 5) meta off_target (still useful)
        # -----------------------------
        off = compute_off_target(target_ipa, asr_ipa)
        is_off_target = bool(off.get("is_off_target"))

        # -----------------------------
        # 6) VowelFinal detector only (script-based)
        # IMPORTANT: detect_vowelfinal_error must return Addition-style fields:
        # final_label + severity (canonical) + type etc.
        # -----------------------------
        vf_res = detect_vowelfinal_error(
            target_word,
            asr_ipa,
            llm=None,                 # ✅ disable LLM explicitly
            off_target=off,           # ✅ pass off_target
            correct_rate=correct_rate # ✅ pass correct_rate
        )

        check_results = {
            "meta": {
                "off_target": off,
                "enabled_checks": ["vowel_final"],
                "pattern": pattern,
                "category": category,
                "llm_model_repo": None,
                "llm_model_file": None,
            },
            "vowel_final": vf_res,
        }

        # -----------------------------
        # 7) status (match Addition endpoint)
        # -----------------------------
        ipa_match = asr_ipa.strip() == target_ipa.strip()
        status = "correct" if ipa_match else "wrong"

        # -----------------------------
        # 8) severity (only when slp)
        # -----------------------------
        vf_final = str((vf_res or {}).get("final_label") or "").strip().lower()
        severity_value = (vf_res or {}).get("severity") if vf_final == "slp" else None
        if severity_value not in ("slight", "middle", "high"):
            severity_value = None

        # -----------------------------
        # 9) DB save
        # -----------------------------
        record_id = save_records_analysis_v2(
            user_id=current_user_id,
            language="Chinese",
            test_type=category,
            category=pattern,
            target_word=target_word,
            target_ipa=target_ipa,
            asr_ipa=asr_ipa,
            asr_word=asr_word,
            correct_rate=correct_rate,
            check_results=check_results,
            difficulty_level=difficulty_level,
            score=score_points,
            severity=severity_value,
        )

        # -----------------------------
        # 10) add points
        # -----------------------------
        if score_points > 0:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "UPDATE patients SET integral = COALESCE(integral,0) + ? WHERE id=?",
                (score_points, current_user_id),
            )
            conn.commit()
            conn.close()

        # -----------------------------
        # 11) return
        # -----------------------------
        return JSONResponse({
            "target_word": target_word,
            "asr_ipa": asr_ipa,
            "revise_ipa": target_ipa,
            "asr_word": asr_word,
            "correct_rate": correct_rate,
            "status": status,
            "check_results": check_results,
            "difficulty_level": difficulty_level,
            "severity": severity_value,
            "is_off_target": is_off_target,
            "score": score_points,
            "record_id": record_id,

            # Prefer vowel_final.patient_explain if present (same pattern as Addition/Tone/Omission)
            "patient_explain": (
                (check_results.get("vowel_final") or {}).get("patient_explain")
                or patient_friendly_explain(check_results)
            ),
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"VowelFinal analysis failed: {str(e)}")
    
# ------------------------------
# API: 分析 InitialConsonant 錯誤
# ------------------------------
from fastapi import UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse

def _pinyin_syllable_list(s: str) -> list[str]:
    s = (s or "").strip().lower().replace("/", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.split() if s else []

def compute_off_target(target_ipa: str, asr_ipa: str) -> dict:
    tgt = _pinyin_syllable_list(target_ipa)
    asr = _pinyin_syllable_list(asr_ipa)
    cr = calc_correct_rate(asr_ipa, target_ipa)

    # 你可按實測調整
    is_off = (cr <= 30 and abs(len(tgt) - len(asr)) >= 1) or (cr == 0)

    return {
        "is_off_target": bool(is_off),
        "correct_rate": cr,
        "target_syllables": len(tgt),
        "asr_syllables": len(asr),
        "syllable_gap": abs(len(tgt) - len(asr)),
    }

# --- EDITED: InitialConsonant endpoint to match Addition process ---
# Key changes vs your current InitialConsonant:
# 1) Run InitialConsonant detector ONLY (like Addition runs addition-only)
# 2) status based on IPA match (same as Addition), NOT "any error or off_target"
# 3) severity ONLY when detector final_label == "slp" (same as Addition)
# 4) subtype Form(...) is treated as legacy/ignored (same as Addition)
# 5) check_results.meta.enabled_checks == ["initial"]

@app.post("/analyze_initial_consonant")
async def analyze_initial_consonant(
    file: UploadFile = File(...),
    target_word: str = Form(...),
    username: str = Form("guest"),
    subtype: int = Form(0),  # legacy; ignored (match Addition behavior)
    category: str = Form("InitialConsonant"),
    pattern: str = Form("InitialConsonant_error"),
    checks: str = Form(""),  # legacy; ignored (initial-only)
    current_user_id: str = Depends(get_current_user_id),
    difficulty_level: str = Form("小學"),
):
    """
    ✅ InitialConsonant-only (script) version — aligned to /analyze_addition

    - Only run InitialConsonant detector (rule-based; no LLM).
      final_label: no_problem | slp | 亂答
      type (subtype): e.g. 声母省略 | 声母添加 | 发音部位错误 | 声母替代  (only if slp)
      severity: slight | middle | high                                 (only if slp)

    - Still compute:
        correct_rate (accuracy, 0~100)
        score (0~10) = round(correct_rate / 10)

    - status:
        correct if asr_ipa == target_ipa
        wrong   otherwise

    - severity:
        only from initial.severity AND only when final_label == slp
    """
    try:
        # -----------------------------
        # 1) read audio
        # -----------------------------
        content = await file.read()

        # -----------------------------
        # 2) ASR
        # -----------------------------
        asr_result = transcribe_bytes(
            audio_bytes=content,
            filename=file.filename,
            target_word=target_word,
            difficulty_level=difficulty_level,
        )

        # -----------------------------
        # 3) results
        # -----------------------------
        asr_ipa = asr_result.get("asr_ipa", "") or ""
        target_ipa = (asr_result.get("target_ipa", "") or "").replace("/", "") or ""

        asr_word = asr_result.get("asr_word", "") or ""
        if not asr_word:
            if asr_ipa.strip() == target_ipa.strip():
                asr_word = target_word
            else:
                asr_word = pinyin_to_chinese_diff(asr_ipa, target_ipa, target_word)

        # -----------------------------
        # 4) accuracy + score points
        # -----------------------------
        correct_rate = calc_correct_rate(asr_ipa, target_ipa)  # 0~100
        score_points = int(round(correct_rate / 10))           # 0~10

        # -----------------------------
        # 5) meta off_target (still useful)
        # -----------------------------
        off = compute_off_target(target_ipa, asr_ipa)
        is_off_target = bool(off.get("is_off_target"))

        # -----------------------------
        # 6) InitialConsonant detector only (script-based)
        # IMPORTANT: this requires your detect_initial_consonant_error
        # to return a dict that includes final_label + severity like Addition does.
        # If your current detect_initial_consonant_error does NOT provide these fields,
        # see the notes below ("What may be different vs Addition").
        # -----------------------------
        init_res = detect_initial_consonant_error(
            target_word,
            asr_ipa,
            llm=None,          # ✅ disable LLM explicitly (if supported)
            off_target=off,    # ✅ pass off_target (if supported)
            correct_rate=correct_rate,  # ✅ pass correct_rate (if supported)
        )

        check_results = {
            "meta": {
                "off_target": off,
                "enabled_checks": ["initial"],
                "pattern": pattern,
                "category": category,
                "llm_model_repo": None,
                "llm_model_file": None,
            },
            "initial": init_res,
        }

        # -----------------------------
        # 7) status (match Addition)
        # -----------------------------
        ipa_match = asr_ipa.strip() == target_ipa.strip()
        status = "correct" if ipa_match else "wrong"

        # -----------------------------
        # 8) severity (only when slp)
        # -----------------------------
        init_res = check_results.get("initial") or {}
        init_final = str(init_res.get("final_label") or "").strip().lower()

        severity_value = init_res.get("severity", None) if init_final == "slp" else None
        if severity_value not in ("slight", "middle", "high"):
            severity_value = None

        # -----------------------------
        # 9) DB save
        # -----------------------------
        record_id = save_records_analysis_v2(
            user_id=current_user_id,
            language="Chinese",
            test_type=category,
            category=pattern,
            target_word=target_word,
            target_ipa=target_ipa,
            asr_ipa=asr_ipa,
            asr_word=asr_word,
            correct_rate=correct_rate,
            check_results=check_results,
            difficulty_level=difficulty_level,
            score=score_points,
            severity=severity_value,
        )

        # -----------------------------
        # 10) add points
        # -----------------------------
        if score_points > 0:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "UPDATE patients SET integral = COALESCE(integral,0) + ? WHERE id=?",
                (score_points, current_user_id),
            )
            conn.commit()
            conn.close()

        # -----------------------------
        # 11) return
        # -----------------------------
        return JSONResponse({
            "target_word": target_word,
            "asr_ipa": asr_ipa,
            "revise_ipa": target_ipa,
            "asr_word": asr_word,
            "correct_rate": correct_rate,
            "status": status,
            "check_results": check_results,
            "difficulty_level": difficulty_level,
            "severity": severity_value,
            "is_off_target": is_off_target,
            "score": score_points,
            "record_id": record_id,

            # Prefer initial.patient_explain if present (same pattern as Addition)
            "patient_explain": (
                (check_results.get("initial") or {}).get("patient_explain")
                or patient_friendly_explain(check_results)
            ),
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"InitialConsonant analysis failed: {str(e)}")


# ------------------------------
# Addition API
# ------------------------------
@app.post("/analyze_addition")
async def analyze_addition(
    file: UploadFile = File(...),
    target_word: str = Form(...),
    username: str = Form("guest"),
    subtype: int = Form(0),  # legacy; ignored
    category: str = Form("Addition"),
    pattern: str = Form("Addition_error"),
    checks: str = Form(""),  # legacy; ignored (addition-only)
    current_user_id: str = Depends(get_current_user_id),
    difficulty_level: str = Form("小學"),
):
    """
    ✅ Addition-only (script) version

    - Only run Addition detector (rule-based; no LLM).
      final_label: no_problem | slp | 亂答
      type (subtype): 声母添加 | 韵母添加 | 韵尾添加 | 插音  (only if slp)
      severity: slight | middle | high                 (only if slp)

    - Still compute:
        correct_rate (accuracy, 0~100)
        score (0~10) = round(correct_rate / 10)

    - status:
        wrong if addition.final_label in {slp, 亂答}
        correct if addition.final_label == no_problem

    - severity:
        only from addition.severity AND only when final_label == slp
    """
    try:
        # -----------------------------
        # 1) read audio
        # -----------------------------
        content = await file.read()

        # -----------------------------
        # 2) ASR
        # -----------------------------
        asr_result = transcribe_bytes(
            audio_bytes=content,
            filename=file.filename,
            target_word=target_word,
            difficulty_level=difficulty_level,
        )

        # -----------------------------
        # 3) results
        # -----------------------------
        asr_ipa = asr_result.get("asr_ipa", "") or ""
        target_ipa = (asr_result.get("target_ipa", "") or "").replace("/", "") or ""

        asr_word = asr_result.get("asr_word", "") or ""
        if not asr_word:
            if asr_ipa.strip() == target_ipa.strip():
                asr_word = target_word
            else:
                asr_word = pinyin_to_chinese_diff(asr_ipa, target_ipa, target_word)

        # -----------------------------
        # 4) accuracy + score points
        # -----------------------------
        correct_rate = calc_correct_rate(asr_ipa, target_ipa)     # ✅ accuracy (0~100)
        score_points = int(round(correct_rate / 10))              # ✅ 0~10

        # -----------------------------
        # 5) meta off_target (still useful)
        # -----------------------------
        off = compute_off_target(target_ipa, asr_ipa)
        is_off_target = bool(off.get("is_off_target"))

        # -----------------------------
        # 6) Addition detector only (script-based)
        # -----------------------------
        add_res = detect_addition_error_1(
            target_word,
            asr_ipa,
            llm=None,                 # ✅ disable LLM explicitly
            off_target=off,
            correct_rate=correct_rate,
        )

        check_results = {
            "meta": {
                "off_target": off,
                "enabled_checks": ["addition"],
                "pattern": pattern,
                "category": category,
                "llm_model_repo": None,
                "llm_model_file": None,
            },
            "addition": add_res,
        }

        # -----------------------------
        # 7) status (addition-only)
        # -----------------------------
        ipa_match = asr_ipa.strip() == target_ipa.strip()
        status = "correct" if ipa_match else "wrong"

        # -----------------------------
        # 8) severity (only when slp)
        # -----------------------------
        add_res = check_results.get("addition") or {}
        add_final = str(add_res.get("final_label") or "").strip().lower()

        severity_value = add_res.get("severity", None) if add_final == "slp" else None
        if severity_value not in ("slight", "middle", "high"):
            severity_value = None


        # -----------------------------
        # 9) DB save
        # -----------------------------
        record_id = save_records_analysis_v2(
            user_id=current_user_id,
            language="Chinese",
            test_type=category,
            category=pattern,
            target_word=target_word,
            target_ipa=target_ipa,
            asr_ipa=asr_ipa,
            asr_word=asr_word,
            correct_rate=correct_rate,
            check_results=check_results,
            difficulty_level=difficulty_level,
            score=score_points,
            severity=severity_value,
        )

        # -----------------------------
        # 10) add points
        # -----------------------------
        if score_points > 0:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "UPDATE patients SET integral = COALESCE(integral,0) + ? WHERE id=?",
                (score_points, current_user_id),
            )
            conn.commit()
            conn.close()

        # -----------------------------
        # 11) return
        # -----------------------------
        return JSONResponse({
            "target_word": target_word,
            "asr_ipa": asr_ipa,
            "revise_ipa": target_ipa,
            "asr_word": asr_word,
            "correct_rate": correct_rate,          # ✅ accuracy
            "status": status,
            "check_results": check_results,
            "difficulty_level": difficulty_level,
            "severity": severity_value,
            "is_off_target": is_off_target,
            "score": score_points,
            "record_id": record_id,

            # Prefer addition.patient_explain if present
            "patient_explain": (
                (check_results.get("addition") or {}).get("patient_explain")
                or patient_friendly_explain(check_results)
            ),
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Addition analysis failed: {str(e)}")

# ------------------------------
# Omission API
# ------------------------------
@app.post("/analyze_omission")
async def analyze_omission(
    file: UploadFile = File(...),
    target_word: str = Form(...),
    username: str = Form("guest"),
    subtype: int = Form(0),  # legacy; ignored (match Addition behavior)
    category: str = Form("Omission"),
    pattern: str = Form("Omission_error"),
    checks: str = Form(""),  # legacy; ignored (omission-only)
    current_user_id: str = Depends(get_current_user_id),
    difficulty_level: str = Form("小學"),
):
    """
    ✅ Omission-only (script) version — aligned to /analyze_addition

    - Only run Omission detector (rule-based; no LLM).
      final_label: no_problem | slp | 亂答
      type (subtype): 声母脱落 | 韵母脱落 | 韵尾脱落 | 声调脱落 (only if slp)
      severity: slight | middle | high                                 (only if slp)

    - Still compute:
        correct_rate (accuracy, 0~100)
        score (0~10) = round(correct_rate / 10)

    - status:
        (match current addition endpoint implementation)
        correct if asr_ipa == target_ipa
        wrong   otherwise

    - severity:
        only from omission.severity AND only when final_label == slp
    """
    try:
        # -----------------------------
        # 1) read audio
        # -----------------------------
        content = await file.read()

        # -----------------------------
        # 2) ASR
        # -----------------------------
        asr_result = transcribe_bytes(
            audio_bytes=content,
            filename=file.filename,
            target_word=target_word,
            difficulty_level=difficulty_level,
        )

        # -----------------------------
        # 3) results
        # -----------------------------
        asr_ipa = asr_result.get("asr_ipa", "") or ""
        target_ipa = (asr_result.get("target_ipa", "") or "").replace("/", "") or ""

        asr_word = asr_result.get("asr_word", "") or ""
        if not asr_word:
            if asr_ipa.strip() == target_ipa.strip():
                asr_word = target_word
            else:
                asr_word = pinyin_to_chinese_diff(asr_ipa, target_ipa, target_word)

        # -----------------------------
        # 4) accuracy + score points
        # -----------------------------
        correct_rate = calc_correct_rate(asr_ipa, target_ipa)  # 0~100
        score_points = int(round(correct_rate / 10))           # 0~10

        # -----------------------------
        # 5) meta off_target (still useful)
        # -----------------------------
        off = compute_off_target(target_ipa, asr_ipa)
        is_off_target = bool(off.get("is_off_target"))

        # -----------------------------
        # 6) Omission detector only (script-based)
        # IMPORTANT: detect_omission_error must return Addition-style fields:
        # final_label + severity (canonical) + type (subtype) etc.
        # -----------------------------
        omit_res = detect_omission_error_1(
            target_word,
            asr_ipa,
            llm=None,                 # ✅ disable LLM explicitly (if supported)
            off_target=off,           # ✅ pass off_target (if supported)
            correct_rate=correct_rate # ✅ pass correct_rate (if supported)
        )

        check_results = {
            "meta": {
                "off_target": off,
                "enabled_checks": ["omission"],
                "pattern": pattern,
                "category": category,
                "llm_model_repo": None,
                "llm_model_file": None,
            },
            "omission": omit_res,
        }

        # -----------------------------
        # 7) status (match Addition endpoint current implementation)
        # -----------------------------
        ipa_match = asr_ipa.strip() == target_ipa.strip()
        status = "correct" if ipa_match else "wrong"

        # -----------------------------
        # 8) severity (only when slp)
        # -----------------------------
        omit_res = check_results.get("omission") or {}
        omit_final = str(omit_res.get("final_label") or "").strip().lower()

        severity_value = omit_res.get("severity", None) if omit_final == "slp" else None
        if severity_value not in ("slight", "middle", "high"):
            severity_value = None

        # -----------------------------
        # 9) DB save
        # -----------------------------
        record_id = save_records_analysis_v2(
            user_id=current_user_id,
            language="Chinese",
            test_type=category,
            category=pattern,
            target_word=target_word,
            target_ipa=target_ipa,
            asr_ipa=asr_ipa,
            asr_word=asr_word,
            correct_rate=correct_rate,
            check_results=check_results,
            difficulty_level=difficulty_level,
            score=score_points,
            severity=severity_value,
        )

        # -----------------------------
        # 10) add points
        # -----------------------------
        if score_points > 0:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "UPDATE patients SET integral = COALESCE(integral,0) + ? WHERE id=?",
                (score_points, current_user_id),
            )
            conn.commit()
            conn.close()

        # -----------------------------
        # 11) return
        # -----------------------------
        return JSONResponse({
            "target_word": target_word,
            "asr_ipa": asr_ipa,
            "revise_ipa": target_ipa,
            "asr_word": asr_word,
            "correct_rate": correct_rate,
            "status": status,
            "check_results": check_results,
            "difficulty_level": difficulty_level,
            "severity": severity_value,
            "is_off_target": is_off_target,
            "score": score_points,
            "record_id": record_id,

            # Prefer omission.patient_explain if present (same pattern as Addition)
            "patient_explain": (
                (check_results.get("omission") or {}).get("patient_explain")
                or patient_friendly_explain(check_results)
            ),
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Omission analysis failed: {str(e)}")


# ------------------------------
# Tone API
# ------------------------------

@app.post("/analyze_tone")
async def analyze_tone(
    file: UploadFile = File(...),
    target_word: str = Form(...),
    username: str = Form("guest"),
    subtype: int = Form(0),          # legacy; ignored (match Addition behavior)
    category: str = Form("Tone"),
    pattern: str = Form("Tone_error"),
    checks: str = Form(""),          # legacy; ignored (tone-only)
    current_user_id: str = Depends(get_current_user_id),
    difficulty_level: str = Form("小學"),
):
    """
    ✅ Tone-only (script) version — aligned to /analyze_addition & /analyze_omission

    - Only run Tone detector (rule-based; no LLM).
      final_label: no_problem | slp | 亂答
      type: 声调替代 | 声调错位 | 轻声错误 (only if slp)
      severity: slight | middle | high               (only if slp)

    - Still compute:
        correct_rate (accuracy, 0~100)
        score (0~10) = round(correct_rate / 10)

    - status:
        correct if asr_ipa == target_ipa
        wrong   otherwise

    - severity:
        only from tone.severity AND only when final_label == slp
    """
    try:
        # -----------------------------
        # 1) read audio
        # -----------------------------
        content = await file.read()

        # -----------------------------
        # 2) ASR
        # -----------------------------
        asr_result = transcribe_bytes(
            audio_bytes=content,
            filename=file.filename,
            target_word=target_word,
            difficulty_level=difficulty_level,
        )

        # -----------------------------
        # 3) results
        # -----------------------------
        asr_ipa = asr_result.get("asr_ipa", "") or ""
        target_ipa = (asr_result.get("target_ipa", "") or "").replace("/", "") or ""

        asr_word = asr_result.get("asr_word", "") or ""
        if not asr_word:
            if asr_ipa.strip() == target_ipa.strip():
                asr_word = target_word
            else:
                asr_word = pinyin_to_chinese_diff(asr_ipa, target_ipa, target_word)

        # -----------------------------
        # 4) accuracy + score points
        # -----------------------------
        correct_rate = calc_correct_rate(asr_ipa, target_ipa)  # 0~100
        score_points = int(round(correct_rate / 10))           # 0~10

        # -----------------------------
        # 5) meta off_target (still useful)
        # -----------------------------
        off = compute_off_target(target_ipa, asr_ipa)
        is_off_target = bool(off.get("is_off_target"))

        # -----------------------------
        # 6) Tone detector only (script-based)
        # IMPORTANT: detect_tone_error must return Addition-style fields:
        # final_label + severity (canonical) + type etc.
        # -----------------------------
        tone_res = detect_tone_error(
            target_word,
            asr_ipa,
            llm=None,                 # ✅ disable LLM explicitly (if supported)
            off_target=off,           # ✅ pass off_target
            correct_rate=correct_rate # ✅ pass correct_rate
        )

        check_results = {
            "meta": {
                "off_target": off,
                "enabled_checks": ["tone"],
                "pattern": pattern,
                "category": category,
                "llm_model_repo": None,
                "llm_model_file": None,
            },
            "tone": tone_res,
        }

        # -----------------------------
        # 7) status (match Addition endpoint)
        # -----------------------------
        ipa_match = asr_ipa.strip() == target_ipa.strip()
        status = "correct" if ipa_match else "wrong"

        # -----------------------------
        # 8) severity (only when slp)
        # -----------------------------
        tone_final = str((tone_res or {}).get("final_label") or "").strip().lower()
        severity_value = (tone_res or {}).get("severity") if tone_final == "slp" else None
        if severity_value not in ("slight", "middle", "high"):
            severity_value = None

        # -----------------------------
        # 9) DB save
        # -----------------------------
        record_id = save_records_analysis_v2(
            user_id=current_user_id,
            language="Chinese",
            test_type=category,
            category=pattern,
            target_word=target_word,
            target_ipa=target_ipa,
            asr_ipa=asr_ipa,
            asr_word=asr_word,
            correct_rate=correct_rate,
            check_results=check_results,
            difficulty_level=difficulty_level,
            score=score_points,
            severity=severity_value,
        )

        # -----------------------------
        # 10) add points
        # -----------------------------
        if score_points > 0:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "UPDATE patients SET integral = COALESCE(integral,0) + ? WHERE id=?",
                (score_points, current_user_id),
            )
            conn.commit()
            conn.close()

        # -----------------------------
        # 11) return
        # -----------------------------
        return JSONResponse({
            "target_word": target_word,
            "asr_ipa": asr_ipa,
            "revise_ipa": target_ipa,
            "asr_word": asr_word,
            "correct_rate": correct_rate,
            "status": status,
            "check_results": check_results,
            "difficulty_level": difficulty_level,
            "severity": severity_value,
            "is_off_target": is_off_target,
            "score": score_points,
            "record_id": record_id,

            # Prefer tone.patient_explain if present (same pattern as Addition/Omission)
            "patient_explain": (
                (check_results.get("tone") or {}).get("patient_explain")
                or patient_friendly_explain(check_results)
            ),
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Tone analysis failed: {str(e)}")

# ------------------------------
# Distortion API
# ------------------------------


@app.post("/analyze_distortion")
async def analyze_distortion(
    file: UploadFile = File(...),
    target_word: str = Form(...),
    username: str = Form("guest"),
    subtype: int = Form(0),             # legacy; ignored (distortion uses bank subtype keys)
    category: str = Form("Distortion"),
    pattern: str = Form("Distortion_error"),
    checks: str = Form(""),             # legacy; ignored (distortion-only)
    current_user_id: str = Depends(get_current_user_id),
    difficulty_level: str = Form("小學"),
):
    """
    ✅ Distortion-only version (aligned with analyze_addition process)

    - Only run Distortion detector (bank-based; no LLM).
      final_label: no_problem | slp | 亂答
      type (subtype): Distortion.json key (e.g., 齿龈摩擦音歪曲 / 塞擦音歪曲) (only if slp)
      severity: slight | middle | high (only if slp; subtype-specific)

    - Still compute:
        correct_rate (accuracy, 0~100)
        score (0~10) = round(correct_rate / 10)

    - status (same as addition page for consistency):
        correct if asr_ipa == target_ipa else wrong

    - severity:
        only from distortion.severity AND only when final_label == slp
    """
    try:
        # -----------------------------
        # 1) read audio
        # -----------------------------
        content = await file.read()

        # -----------------------------
        # 2) ASR
        # -----------------------------
        asr_result = transcribe_bytes(
            audio_bytes=content,
            filename=file.filename,
            target_word=target_word,
            difficulty_level=difficulty_level,
        )

        # -----------------------------
        # 3) results
        # -----------------------------
        asr_ipa = asr_result.get("asr_ipa", "") or ""
        target_ipa = (asr_result.get("target_ipa", "") or "").replace("/", "") or ""

        asr_word = asr_result.get("asr_word", "") or ""
        if not asr_word:
            if asr_ipa.strip() == target_ipa.strip():
                asr_word = target_word
            else:
                asr_word = pinyin_to_chinese_diff(asr_ipa, target_ipa, target_word)

        # -----------------------------
        # 4) accuracy + score points
        # -----------------------------
        correct_rate = calc_correct_rate(asr_ipa, target_ipa)     # 0~100
        score_points = int(round(correct_rate / 10))              # 0~10

        # -----------------------------
        # 5) meta off_target
        # -----------------------------
        off = compute_off_target(target_ipa, asr_ipa)
        is_off_target = bool(off.get("is_off_target"))

        # -----------------------------
        # 6) Distortion detector only
        # -----------------------------
        dis_res = detect_distortion_error(
            target_word,
            asr_ipa,
            llm=None,                 # keep signature compatible; not used
            off_target=off,
            correct_rate=correct_rate,
        )

        check_results = {
            "meta": {
                "off_target": off,
                "enabled_checks": ["distortion"],
                "pattern": pattern,
                "category": category,
                "llm_model_repo": None,
                "llm_model_file": None,
            },
            "distortion": dis_res,
        }

        # -----------------------------
        # 7) status (distortion-only; align to addition)
        # -----------------------------
        ipa_match = asr_ipa.strip() == target_ipa.strip()
        status = "correct" if ipa_match else "wrong"

        # -----------------------------
        # 8) severity (only when slp)
        # -----------------------------
        dis_final = str((dis_res or {}).get("final_label") or "").strip().lower()

        severity_value = (dis_res or {}).get("severity", None) if dis_final == "slp" else None
        if severity_value not in ("slight", "middle", "high"):
            severity_value = None

        # -----------------------------
        # 9) DB save
        # -----------------------------
        record_id = save_records_analysis_v2(
            user_id=current_user_id,
            language="Chinese",
            test_type=category,
            category=pattern,
            target_word=target_word,
            target_ipa=target_ipa,
            asr_ipa=asr_ipa,
            asr_word=asr_word,
            correct_rate=correct_rate,
            check_results=check_results,
            difficulty_level=difficulty_level,
            score=score_points,
            severity=severity_value,
        )

        # -----------------------------
        # 10) add points
        # -----------------------------
        if score_points > 0:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "UPDATE patients SET integral = COALESCE(integral,0) + ? WHERE id=?",
                (score_points, current_user_id),
            )
            conn.commit()
            conn.close()

        # -----------------------------
        # 11) return
        # -----------------------------
        return JSONResponse({
            "target_word": target_word,
            "asr_ipa": asr_ipa,
            "revise_ipa": target_ipa,
            "asr_word": asr_word,
            "correct_rate": correct_rate,
            "status": status,
            "check_results": check_results,
            "difficulty_level": difficulty_level,
            "severity": severity_value,
            "is_off_target": is_off_target,
            "score": score_points,
            "record_id": record_id,

            # Prefer distortion.patient_explain if exists (old style), else fallback
            "patient_explain": (
                (check_results.get("distortion") or {}).get("patient_explain")
                or patient_friendly_explain(check_results)
            ),
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Distortion analysis failed: {str(e)}")

def normalize_practice_language(lang: str | None) -> str:
    """
    practice_progress.language 的值只用: 'en' | 'cn'
    - default: en
    """
    s = (lang or "").strip().lower()
    if s in ("cn", "zh", "zh-cn", "chinese", "china"):
        return "cn"
    return "en"

# ------------------------------
# CORS 設定
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發階段允許所有，正式環境建議改成 ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# User APIs (JSON)
# ------------------------------
class UserRegister(BaseModel):
    username: str
    password: str
    name: Optional[str] = None
    age: Optional[int] = None
    email: Optional[str] = None
    language: Optional[str] = "en"

# ------------------------------
# API: 取得 InitialConsonant 詞語
# ------------------------------
@app.get("/get_word_from_initial_consonant")
async def get_word_from_initial_consonant(errorType: str = Query(..., description="Initial consonant error type")):
    """
    從 InitialConsonant.json 根據 errorType 抽一個詞回傳
    - errorType: "声母省略" / "声母替代" / "声母添加" / "发音部位错误"
    """
    try:
        if not errorType or errorType not in InitialConsonant:
            raise HTTPException(status_code=400, detail="Invalid errorType")

        word = random.choice(InitialConsonant[errorType])
        return {"word": word}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取詞失敗: {str(e)}")



# ===========================
# ✅ Updated Register Function (with email verify)
# ===========================
@app.post("/auth/register")
def register(data: UserRegister):
    conn = get_db()
    cur = conn.cursor()

    # ==========================
    # ✅ 檢查用戶名是否已存在
    # ==========================
    cur.execute("SELECT 1 FROM patients WHERE username=?", (data.username,))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="❌ Username already exists, please choose another one")

    # ==========================
    # ✅ 檢查 Email 是否已存在
    # ==========================
    if data.email:
        cur.execute("SELECT 1 FROM patients WHERE email=?", (data.email,))
        if cur.fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail="❌ Email already registered, please use another email")

    # ==========================
    # ✅ 產生 6 位驗證碼
    # ==========================
    verify_code = str(random.randint(100000, 999999))

    # ==========================
    # ✅ 寄送驗證信
    # ==========================
    try:
        sender = "aaa2025819@gmail.com"
        smtp_host = "smtp.gmail.com"
        smtp_port = 587
        smtp_user = "aaa2025819@gmail.com"
        smtp_pass = "ssjuayuhpfugwltp"   # ⚠️ Gmail App Password

        subject = "ASR Verification Code"
        body = f"""
Hello {data.username},

Your verification code is: {verify_code}

Please copy this code into the website to activate your account.

Thank you,
ASR+LLM System
"""

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = sender
        msg["To"] = data.email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(sender, [data.email], msg.as_string())

    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Email send failed: {e}")

    # ==========================
    # ✅ 建立新帳號（verify=0）
    # ==========================
    patient_id = str(uuid.uuid4())
    email_value = data.email or ""
    language_value = data.language or "en"

    cur.execute("""
        INSERT INTO patients
        (id, username, password_hash, name, age, email, language, integral, verify)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
    """, (
        patient_id,
        data.username,
        hash_password(data.password),
        data.name,
        data.age,
        email_value,
        language_value,
        0
    ))

    # ==========================
    # ✅ 儲存驗證碼
    # ==========================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS email_verifies (
          username TEXT PRIMARY KEY,
          code TEXT,
          created_at TIMESTAMP
        )
    """)
    cur.execute("REPLACE INTO email_verifies (username, code, created_at) VALUES (?, ?, ?)",
                (data.username, verify_code, datetime.now()))

    conn.commit()
    conn.close()

    return {
        "msg": f"📩 Verification code sent to {data.email}. Please check your inbox.",
        "username": data.username,
        "email": data.email
    }

class UserLogin(BaseModel):
    email: str
    password: str

@app.get("/api/leaderboard")
def get_leaderboard(limit: int = 100):
    """
    📊 取得積分排行榜前 N 名 (排除 admin 帳號)
    """
    conn = get_db()
    cur = conn.cursor()
    # 🚫 排除管理者
    cur.execute("""
        SELECT username, integral
        FROM patients
        WHERE username IS NOT NULL
          AND username NOT IN ('admin', 'bbb')
        ORDER BY integral DESC
        LIMIT ?
    """, (limit,))

    rows = cur.fetchall()
    conn.close()

    # 組成 JSON
    leaderboard = [
        {"rank": i + 1, "username": r[0], "integral": r[1] or 0}
        for i, r in enumerate(rows)
    ]
    return {"leaderboard": leaderboard}

@app.post("/auth/login")
def login(
    request: Request,
    data: UserLogin,
    response: Response,
    send_cookie: bool = Query(SEND_USERNAME_COOKIE_DEFAULT, description="是否下發 email cookie"),
):
    # 🔒 Admin still limited to localhost (but now identified by email)
    if data.email == f"{ADMIN_LOCAL_ONLY}@example.com":
        client_host = request.client.host
        if client_host not in ["127.0.0.1", "::1"]:
            raise HTTPException(status_code=403, detail="Admin account can only login from localhost")

    # ✅ 查詢 email 而非 username
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash FROM patients WHERE email=?", (data.email,))
    row = cur.fetchone()
    conn.close()

    if not row or not verify_password(data.password, row[2]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_id = row[0]
    username = row[1]

    expire = datetime.utcnow() + timedelta(hours=12)
    token_data = {"sub": user_id, "exp": expire}
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

    # 🔑 只有 send_cookie=True 時才送
    if send_cookie:
        response.set_cookie(
            key="email",
            value=data.email,
            httponly=False,  # 前端 JS 仍可存取
            samesite="Lax"
        )

    # ✅ 取得使用者頭像路徑並設置 Cookie
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT user_picture FROM patients WHERE id=?", (user_id,))
    row = cur.fetchone()
    conn.close()

    picture_url = None
    if row and row[0]:
        picture_url = f"/{row[0]}"
        response.set_cookie(
            key="user_picture_url",
            value=picture_url,
            path="/",
            samesite="Lax",
            secure=True
        )

    return {
        "msg": "Login successful",
        "email": data.email,
        "username": username,
        "id": user_id,
        "token": token,
        "picture": picture_url
    }


# ✅ 新增：Change Password API
class ChangePasswordRequest(BaseModel):
    username: str
    old_password: str
    new_password: str


@app.post("/auth/change-password")
def change_password(data: ChangePasswordRequest):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM patients WHERE username=?", (data.username,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(data.old_password, row[0]):
        conn.close()
        raise HTTPException(status_code=400, detail="Old password is incorrect")

    cur.execute("UPDATE patients SET password_hash=? WHERE username=?",
                (hash_password(data.new_password), data.username))
    conn.commit()
    conn.close()
    return {"msg": "Password updated successfully"}

# ✅ 新增：Update Profile API
class UpdateProfileRequest(BaseModel):
    username: str
    name: Optional[str] = None
    age: Optional[int] = None
    diagnosis: Optional[str] = None
    email: Optional[str] = None
    language: Optional[str] = None
    integral: Optional[int] = None
    verify: Optional[int] = None
    old_password: Optional[str] = None
    new_password: Optional[str] = None

# ✅ 新增：Update Profile API
class UpdateProfileRequest(BaseModel):
    username: str
    name: Optional[str] = None
    age: Optional[int] = None
    diagnosis: Optional[str] = None
    email: Optional[str] = None
    language: Optional[str] = None
    integral: Optional[int] = None
    verify: Optional[int] = None
    old_password: Optional[str] = None
    new_password: Optional[str] = None


@app.post("/auth/update-profile")
def update_profile(data: UpdateProfileRequest):
    """
    根據傳入欄位動態更新患者資料。
    - 沒有傳入的欄位不會更改。
    - 支援更新 language, email, integral, verify。
    - 修改密碼需同時帶 old_password + new_password。
    """
    conn = get_db()
    cur = conn.cursor()

    # 先確認帳號存在
    cur.execute("SELECT id, password_hash FROM patients WHERE username=?", (data.username,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    user_id, stored_hash = row

    updates = []
    params = []

    # ✅ 動態欄位（只更新有傳進來的）
    if data.name is not None:
        updates.append("name=?")
        params.append(data.name)

    if data.age is not None:
        updates.append("age=?")
        params.append(data.age)

    if data.diagnosis is not None:
        updates.append("diagnosis=?")
        params.append(data.diagnosis)

    if data.email is not None:
        updates.append("email=?")
        params.append(data.email)

    if data.language is not None:
        updates.append("language=?")
        params.append(data.language)

    if data.integral is not None:
        updates.append("integral=?")
        params.append(data.integral)

    if data.verify is not None:
        updates.append("verify=?")
        params.append(data.verify)

    # ✅ 若要修改密碼，必須同時檢查舊密碼
    if data.new_password:
        if not data.old_password or not verify_password(data.old_password, stored_hash):
            conn.close()
            raise HTTPException(status_code=400, detail="Old password incorrect")

        updates.append("password_hash=?")
        params.append(hash_password(data.new_password))

    # 沒有任何項目要更新時
    if not updates:
        conn.close()
        return {"msg": "No fields updated"}

    # ✅ 組 UPDATE 語句
    sql = f"UPDATE patients SET {', '.join(updates)} WHERE username=?"
    params.append(data.username)
    cur.execute(sql, tuple(params))
    conn.commit()

    # ✅ 更新完成後回傳完整資料（你 Setting 頁會用到的欄位）
    cur.execute("""
        SELECT id, username, name, age, diagnosis, email, language, integral, verify, user_picture
        FROM patients WHERE username=?
    """, (data.username,))
    user_row = cur.fetchone()
    conn.close()

    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "msg": "Profile updated successfully",
        "user": {
            "id": user_row[0],
            "username": user_row[1],
            "name": user_row[2],
            "age": user_row[3],
            "diagnosis": user_row[4],
            "email": user_row[5],
            "language": user_row[6],
            "integral": user_row[7],
            "verify": user_row[8],
            "user_picture": user_row[9],
        }
    }
    

@app.post("/get_word_llm1")
async def get_word_llm1(data: WordRequest):
    error_type = data.errorType
    if error_type not in substitution_errors_llm1:
        raise HTTPException(status_code=400, detail=f"Unknown error type: {error_type}")

    word_list = substitution_errors_llm1[error_type]

    # 這裡可以不用 LLM，直接隨機選一個就好
    chosen_word = random.choice(word_list)

    return {"word": chosen_word}
# ------------------------------
# Pronunciation APIs
# ------------------------------
# ------------------------------
# Pronunciation APIs
# ------------------------------
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    target_word: str = Form(...),
    difficulty_level: str = Form("小學")
):
    """
    接收錄音和中文目標詞，傳給 ASR+LLM 分析，回傳結果
    """
    try:
        # 暫存錄音檔
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 🔹 呼叫 ASR 模組 (這裡用 recognize_word → 得到 ASR IPA)
        audio = load_and_denoise(tmp_path)
        asr_ipa = normalize_ipa(recognize_word(audio))
        os.remove(tmp_path)

        # 🔹 查字典裡的目標詞 IPA（如果沒有就用 eng_to_ipa）
        target_ipas = pronunciation_dict.get(target_word.lower(), [ipa.convert(target_word)])
        target_ipa = target_ipas[0]

        # 🔹 判斷是否正確
        status = "correct" if asr_ipa == target_ipa else "wrong"

        # 🔹 這裡簡單模擬 LLM 建議練習詞（實際可改成 generate_practice_word）
        suggest_word = random.choice([target_word, "媽媽", "麻"])  

        result = {
            "target_word": target_word,
            "asr_ipa": asr_ipa,
            "target_ipa": target_ipa,
            "status": status,
            "suggest_word": suggest_word,
            "audio_url": f"/google_tts/{target_word}"
        }

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcribe error: {str(e)}")

#-----------English SLP ------------------------------#

def normalize_difficulty_en(difficulty_level: str) -> str:
    """
    Map legacy values to DB/LLM accepted buckets: primary|secondary|advanced
    """
    s = (difficulty_level or "").strip().lower()
    mapping = {
        "primary_school": "primary",
        "primary": "primary",
        "elementary": "primary",
        "secondary": "secondary",
        "middle": "secondary",
        "advanced": "advanced",
        "advance": "advanced",
    }
    return mapping.get(s, "advanced")

def normalize_ipa_simple(s: str) -> str:
    return (s or "").replace("ˈ", "").replace("ˌ", "").replace("ː", "").strip()

def enforce_word_ipa_consistency(target_word: str, target_ipa: str, asr_word: str, asr_ipa: str):
    tw = (target_word or "").strip()
    aw = (asr_word or "").strip()

    ti = normalize_ipa_simple(target_ipa)
    ai = normalize_ipa_simple(asr_ipa)

    # A) word相同但IPA不同 -> 强制IPA一致
    if tw and aw and tw.lower() == aw.lower() and ti and ai and ti != ai:
        asr_ipa = target_ipa
        ai = ti

    # B) IPA相同但word不同 -> 强制word一致
    if ti and ai and ti == ai and tw and aw and tw.lower() != aw.lower():
        asr_word = target_word

    return asr_word, asr_ipa
    
@app.post("/analyze_addition_words")
async def analyze_addition_words(
    file: UploadFile = File(...),
    target_word: str = Form(...),

    # 前端傳
    test_type: str = Form("addition"),
    category: str = Form(...),

    pattern: str = Form(None),

    # 可選：LLM 參數（不傳就用預設）
    num_cases: Optional[int] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    temperature: float = Form(0.2),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.05),

    current_user_id: str = Depends(get_current_user_id),
):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files supported")

    content = await file.read()

    # 1) ASR
    asr = transcribe_bytes_en_full(content, target_word=target_word)
    target_ipa = asr.get("target_ipa", "")
    asr_ipa = asr.get("asr_ipa", "")
    asr_word = asr.get("asr_word") or "N/A"

    # 2) normalize difficulty
    diff = infer_en_difficulty_from_user_id(current_user_id)
    
    # 3) build required fields for analyze_record_payload
    req_id = str(uuid.uuid4())
    req_date = datetime.now().strftime("%Y-%m-%d")

    payload = analyze_record_payload(
        id=req_id,
        user_id=current_user_id,   # ✅ 用登入者 id
        date=req_date,
        target_word=target_word,
        target_ipa=target_ipa,
        asr_word=asr_word,
        asr_ipa=asr_ipa,
        difficulty=diff,           # ✅ 用你 normalize 過的 diff
        language="en",
        test_type=test_type,
        category=category,
        num_cases=num_cases,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    # optional: keep frontend pattern
    if pattern and isinstance(payload.get("check_results"), dict):
        meta = payload["check_results"].get("meta") or {}
        meta["pattern"] = pattern
        payload["check_results"]["meta"] = meta

    # 4) DB save
    record_id = save_records_analysis_v2(
        user_id=current_user_id,
        language="English",
        test_type=payload["test_type"],
        category=payload["category"],
        target_word=payload["target_word"],
        target_ipa=payload["target_ipa"],
        asr_ipa=payload["asr_ipa"],
        asr_word=payload["asr_word"],
        correct_rate=payload["correct_rate"],
        check_results=payload["check_results"],
        difficulty_level=payload["difficulty"],
        score=payload["score"],
        severity=payload["severity"],
    )

    out = dict(payload)
    out["record_id"] = record_id
    out["revise_ipa"] = payload.get("target_ipa")
    out["addition_check"] = (payload.get("check_results") or {}).get("addition")
    return JSONResponse(out)

@app.post("/analyze_substitution_words")
async def analyze_substitution_words(
    file: UploadFile = File(...),
    target_word: str = Form(...),

    # 前端傳（跟 addition 一樣）
    test_type: str = Form("substitution"),   # ✅ supertype = substitution
    category: str = Form(...),               # ✅ required (跟 addition 一樣)
    pattern: str = Form(None),

    # 可選：LLM 參數（跟 addition 一樣）
    num_cases: Optional[int] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    temperature: float = Form(0.2),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.05),

    current_user_id: str = Depends(get_current_user_id),
):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files supported")

    content = await file.read()

    # 1) ASR（沿用你 addition_words 用的英文 ASR：transcribe_bytes_en_full）
    asr = transcribe_bytes_en_full(content, target_word=target_word)
    target_ipa = asr.get("target_ipa", "")
    asr_ipa = asr.get("asr_ipa", "")
    asr_word = asr.get("asr_word") or "N/A"

    # 2) normalize difficulty（跟 addition 一樣：由 user_id 推斷）
    diff = infer_en_difficulty_from_user_id(current_user_id)

    # 3) build required fields for analyze_record_payload（跟 addition 一樣）
    req_id = str(uuid.uuid4())
    req_date = datetime.now().strftime("%Y-%m-%d")

    payload = analyze_record_payload(
        id=req_id,
        user_id=current_user_id,
        date=req_date,
        target_word=target_word,
        target_ipa=target_ipa,
        asr_word=asr_word,
        asr_ipa=asr_ipa,
        difficulty=diff,
        language="en",

        # ✅ substitution 這裡要傳 substitution
        test_type=test_type,     # default "substitution"
        category=category,

        num_cases=num_cases,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    # optional: keep frontend pattern（跟 addition 一樣）
    if pattern and isinstance(payload.get("check_results"), dict):
        meta = payload["check_results"].get("meta") or {}
        meta["pattern"] = pattern
        payload["check_results"]["meta"] = meta

    # 4) DB save（✅ 用同一個 save_records_analysis_v2，只存一次）
    record_id = save_records_analysis_v2(
        user_id=current_user_id,
        language="English",
        test_type=payload["test_type"],     # "substitution"
        category=payload["category"],
        target_word=payload["target_word"],
        target_ipa=payload["target_ipa"],
        asr_ipa=payload["asr_ipa"],
        asr_word=payload["asr_word"],
        correct_rate=payload["correct_rate"],
        check_results=payload["check_results"],
        difficulty_level=payload["difficulty"],
        score=payload["score"],
        severity=payload["severity"],
    )

    out = dict(payload)
    out["record_id"] = record_id
    out["revise_ipa"] = payload.get("target_ipa")

    # ✅ substitution 的前端若需要對應欄位，你可以加這個（取決於 analyze_record_payload 的結構）
    out["substitution_check"] = (payload.get("check_results") or {}).get("substitution")

    return JSONResponse(out)

@app.post("/analyze_omission_words")
async def analyze_omission_words(
    file: UploadFile = File(...),
    target_word: str = Form(...),

    # 前端傳（跟 addition 一樣）
    test_type: str = Form("omission"),   # ✅ supertype = omission
    category: str = Form(...),           # ✅ required
    pattern: str = Form(None),

    # 可選：LLM 參數（跟 addition 一樣）
    num_cases: Optional[int] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    temperature: float = Form(0.2),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.05),

    current_user_id: str = Depends(get_current_user_id),
):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files supported")

    content = await file.read()

    # 1) ASR（跟 addition 一樣）
    asr = transcribe_bytes_en_full(content, target_word=target_word)
    target_ipa = asr.get("target_ipa", "")
    asr_ipa = asr.get("asr_ipa", "")
    asr_word = asr.get("asr_word") or "N/A"

    # 2) normalize difficulty（跟 addition 一樣）
    diff = infer_en_difficulty_from_user_id(current_user_id)

    # 3) build required fields for analyze_record_payload（跟 addition 一樣）
    req_id = str(uuid.uuid4())
    req_date = datetime.now().strftime("%Y-%m-%d")

    payload = analyze_record_payload(
        id=req_id,
        user_id=current_user_id,
        date=req_date,
        target_word=target_word,
        target_ipa=target_ipa,
        asr_word=asr_word,
        asr_ipa=asr_ipa,
        difficulty=diff,
        language="en",

        # ✅ omission
        test_type=test_type,   # default "omission"
        category=category,

        num_cases=num_cases,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    # optional: keep frontend pattern（跟 addition 一樣）
    if pattern and isinstance(payload.get("check_results"), dict):
        meta = payload["check_results"].get("meta") or {}
        meta["pattern"] = pattern
        payload["check_results"]["meta"] = meta

    # 4) DB save（✅ 只存這一次）
    record_id = save_records_analysis_v2(
        user_id=current_user_id,
        language="English",
        test_type=payload["test_type"],     # "omission"
        category=payload["category"],
        target_word=payload["target_word"],
        target_ipa=payload["target_ipa"],
        asr_ipa=payload["asr_ipa"],
        asr_word=payload["asr_word"],
        correct_rate=payload["correct_rate"],
        check_results=payload["check_results"],
        difficulty_level=payload["difficulty"],
        score=payload["score"],
        severity=payload["severity"],
    )

    out = dict(payload)
    out["record_id"] = record_id
    out["revise_ipa"] = payload.get("target_ipa")

    # （可選）如果你的 analyze_record_payload 有 omission key，前端要用的話可以加
    out["omission_check"] = (payload.get("check_results") or {}).get("omission")

    return JSONResponse(out)
#-----------English SLP ------------------------------#


@app.get("/history/{username}")
def get_history(username: str):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id FROM patients WHERE username=?", (username,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    patient_id = row[0]

    cur.execute("SELECT id, date FROM sessions WHERE patient_id=? ORDER BY date DESC", (patient_id,))
    sessions = []
    for s_id, s_date in cur.fetchall():
        cur.execute("SELECT word, category, pattern, asr_ipa, revise_ipa, status, audio_path FROM attempts WHERE session_id=?", (s_id,))
        attempts = [
            {
                "word": a[0],
                "category": a[1],
                "pattern": a[2],
                "asr_ipa": a[3],
                "revise_ipa": a[4],
                "status": a[5],
                "audio_path": a[6]
            }
            for a in cur.fetchall()
        ]
        sessions.append({"date": s_date, "attempts": attempts})

    conn.close()
    return {"username": username, "history": sessions}


@app.get("/auth/me")
def get_me(
    username: str = Query(..., alias="username"), 
    simple: bool = Query(False, description="只回傳 is_admin")
):
    token = username
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM patients WHERE id=?", (user_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    patient_id, username = row
    is_admin = username in ADMIN_USERS

    # ✅ 如果 simple=true 就只回傳這個
    if simple:
        conn.close()
        return {
            "id": patient_id,
            "username": username,
            "is_admin": is_admin
        }

    # ✅ 否則照舊回傳 sessions
    cur.execute("SELECT id, date FROM sessions WHERE patient_id=? ORDER BY date DESC", (patient_id,))
    sessions = []
    for s_id, s_date in cur.fetchall():
        cur.execute("""
            SELECT word, category, pattern, asr_ipa, revise_ipa, status, audio_path
            FROM attempts WHERE session_id=?""", (s_id,))
        attempts = [
            {
                "word": a[0],
                "category": a[1],
                "pattern": a[2],
                "asr_ipa": a[3],
                "revise_ipa": a[4],
                "status": a[5],
                "audio_path": a[6]
            }
            for a in cur.fetchall()
        ]
        sessions.append({"date": s_date, "attempts": attempts})

    conn.close()
    return {
        "id": patient_id,
        "username": username,
        "sessions": sessions
    }

@app.post("/auth/upload-picture")
async def upload_user_picture(
    file: UploadFile = File(...),
    token: Optional[str] = Cookie(None),
    response: Response = None
):
    """
    上傳使用者頭像：
    - 檢查登入 Token
    - 檢查副檔名格式 (.png / .jpg / .jpeg)
    - 檢查檔案大小 ( ≤ 1MB)
    - 儲存於 User/<token>/picture/profile.png
    - 更新資料庫 user_picture 欄位
    - 回傳詳細訊息
    """
    try:
        # 🧩 Step 1️⃣ 驗證登入 token
        if not token:
            raise HTTPException(
                status_code=401, 
                detail="❌ 請先登入（cookie 中沒有 token）"
            )

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="❌ Token 無效，未包含使用者 ID。")
        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"❌ Token 錯誤或過期: {str(e)}")

        # 🧩 Step 2️⃣ 檢查檔案名稱
        if not file.filename:
            raise HTTPException(status_code=400, detail="❌ 未收到檔案或檔名為空")

        filename = file.filename.lower()
        if not (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
            raise HTTPException(
                status_code=400,
                detail=f"❌ 檔案格式錯誤 ({filename})，僅允許上傳 PNG / JPG / JPEG"
            )

        # 🧩 Step 3️⃣ 檢查檔案大小
        contents = await file.read()
        file_size_kb = len(contents) / 1024
        max_size_kb = 1000  # 你原本註解寫 300KB，但實際設定 1000KB≈1MB
        if file_size_kb > max_size_kb:
            raise HTTPException(
                status_code=400,
                detail=f"❌ 檔案大小超過限制：{file_size_kb:.1f}KB（最大允許 {max_size_kb:.0f}KB）"
            )

        # 🧩 Step 4️⃣ 建立使用者目錄（❗改成以 user_id 命名）
        try:
            user_dir = os.path.join(USER_ROOT_DIR, "patient", str(user_id))
            picture_dir = os.path.join(user_dir, "picture")
            os.makedirs(picture_dir, exist_ok=True)

            # 刪除舊圖
            for old in os.listdir(picture_dir):
                os.remove(os.path.join(picture_dir, old))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"❌ 無法建立使用者資料夾: {str(e)}")

        # 🧩 Step 5️⃣ 儲存新檔案（這裡不動）
        try:
            ext = ".png" if filename.endswith(".png") else ".jpg"
            file_name = f"profile{ext}"
            save_path = os.path.join(picture_dir, file_name)
            with open(save_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"❌ 儲存圖片失敗: {str(e)}")

        # 🧩 Step 6️⃣ 更新資料庫（❗改為以 user_id 存路徑）
        try:
            conn = get_db()
            cur = conn.cursor()
            relative_path = f"User/patient/{user_id}/picture/{file_name}"
            cur.execute(
                "UPDATE patients SET user_picture=? WHERE id=?",
                (relative_path, user_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"❌ 更新資料庫失敗: {str(e)}")

        # 🧩 Step 7️⃣ 設定 Cookie + 回傳成功訊息（這裡可保留）
        public_url = f"/{relative_path}"
        if response:
            response.set_cookie(
                key="user_picture_url",
                value=public_url,
                httponly=False,
                samesite="Lax",
                secure=True
            )

        return JSONResponse(
            content={
                "msg": f"✅ 上傳成功！檔案大小 {file_size_kb:.1f}KB，已存至 {public_url}",
                "picture_url": public_url
            },
            headers={"Set-Cookie": f"user_picture_url={public_url}; path=/; SameSite=Lax"}
        )

    except HTTPException:
        raise
    except Exception as e:
        # 🧩 捕捉所有未預期錯誤
        raise HTTPException(status_code=500, detail=f"❌ 未知錯誤: {str(e)}")


@app.post("/api/ask-route")
async def ask_route(data: dict):
    try:
        question = data.get("question", "")
        if not question.strip():
            raise HTTPException(status_code=400, detail="缺少問題內容")
        reply = ask_about_route(question)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route QA error: {str(e)}")

@app.post("/api/ask")
async def ask_api(data: dict):
    username = data.get("username", "")
    question = data.get("question", "")

    if not question.strip():
        raise HTTPException(status_code=400, detail="缺少問題內容")

    if username in ADMIN_USERS:   # ✅ 支援多個 admin
        reply = main_chat.generate_text(
            question,
            role_instruction="你是一個助理，可以自由回答使用者任何問題。"
        )
        main_chat.save_message(username, "user", question)
        main_chat.save_message(username, "assistant", reply)
        return {"role": "admin", "reply": reply}

    else:
        # 普通使用者只能問路徑
        if any(kw in question for kw in ["去哪裡", "頁面", "path", "route"]):
            reply = ask_about_route(question)
        else:
            reply = "❌ 抱歉，你沒有權限詢問與路徑無關的問題。"

        main_chat.save_message(username, "user", question)
        main_chat.save_message(username, "assistant", reply)
        return {"role": "user", "reply": reply}

@app.post("/api/chat")
async def chat_api(data: dict = None):
    try:
        if not data or "message" not in data:
            raise HTTPException(status_code=400, detail="Missing message field")

        message = data["message"]
        user_id = data.get("username", "guest")

        # 1️⃣ 存使用者訊息
        main_chat.save_message(user_id, "user", message)

        # 2️⃣ 撈最近 20 條對話
        recent = main_chat.get_recent_messages(user_id, limit=20)
        history_msgs = [{"role": role, "content": text} for role, text in recent]

        # 3️⃣ 加入這次的新問題
        history_msgs.append({"role": "user", "content": message})

        # 4️⃣ 呼叫 LLM
        resp = ollama.chat(
            model="gpt-oss:20b",
            messages=history_msgs,
            options={"num_predict": 500}
        )
        reply = resp["message"]["content"]

        # 5️⃣ 存 AI 回覆
        main_chat.save_message(user_id, "assistant", reply)

        return {"reply": reply}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# ==============================
# 💾 統一存 DB 函式
# ==============================
def save_message_to_db(username: str, role: str, message: str, 
                       aitest: str = None, file_type: str = None, file_path: str = None):
    conn = sqlite3.connect("memory.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            username TEXT,
            role TEXT,
            message TEXT,
            aitest TEXT,
            file_type TEXT,
            file_path TEXT,
            created_at TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    """)
    cur.execute(
        """INSERT INTO messages 
        (id, username, role, message, aitest, created_at, is_active) 
        VALUES (?, ?, ?, ?, ?, ?, 1)""",
        (str(uuid.uuid4()), username, role, message, aitest, datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


# -----------------------------
# 上傳音訊
# -----------------------------
@app.post("/api/chat/upload-audio")
async def chat_audio_api(
    file: UploadFile = File(...),
    username: str = Depends(get_current_username_from_db)
):
    # 1️⃣ 管理員檢查
    if ONLY_ADMIN_UPLOAD and username not in ADMIN_USERS:
        raise HTTPException(status_code=403, detail="❌ 只有管理員可以上傳音訊")

    # 2️⃣ 檔案類型檢查
    mime = magic.from_buffer(await file.read(2048), mime=True)
    await file.seek(0)
    if not mime.startswith("audio/"):
        raise HTTPException(status_code=400, detail=f"❌ 僅支援音訊，上傳類型為: {mime}")

    # 3️⃣ 保存文件
    save_path = os.path.join(UPLOAD_AUDIO_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ 先記錄上傳動作
    save_message_to_db(
        username,
        "system",
        f"🎙️ 音訊已上傳: {file.filename}",
        file_type="audio",
        file_path=save_path
    )

    # 4️⃣ 呼叫 LLM 分析
    reply = main_chat.chat_with_audio(save_path)

    # 5️⃣ 存 AI 回覆
    save_message_to_db(username, "llm", reply)

    return {"reply": reply, "filename": file.filename, "mime": mime}


# -----------------------------
# 上傳圖片
# -----------------------------
@app.post("/api/chat/upload-image")
async def chat_image_api(
    file: UploadFile = File(...),
    username: str = Depends(get_current_username_from_db)
):
    # 1️⃣ 管理員檢查
    if ONLY_ADMIN_UPLOAD and username not in ADMIN_USERS:
        raise HTTPException(status_code=403, detail="❌ 只有管理員可以上傳圖片")

    # 2️⃣ 檔案類型檢查
    mime = magic.from_buffer(await file.read(2048), mime=True)
    await file.seek(0)
    if not mime.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"❌ 僅支援圖片，上傳類型為: {mime}")

    # 3️⃣ 保存文件
    save_path = os.path.join(UPLOAD_IMAGE_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ 先記錄上傳動作
    save_message_to_db(
        username,
        "system",
        f"🖼️ 圖片已上傳: {file.filename}",
        file_type="image",
        file_path=save_path
    )

    # 4️⃣ 呼叫 LLM 分析
    with open(save_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    reply = main_chat.chat_with_image(b64)

    # 5️⃣ 存 AI 回覆
    save_message_to_db(username, "llm", reply)

    return {"reply": reply, "filename": file.filename, "mime": mime}


# -----------------------------
# 上傳文件（單純存檔）
# -----------------------------
@app.post("/upload/document")
async def upload_document(
    file: UploadFile = File(...),
    username: str = Depends(get_current_username_from_db)
):
    if ONLY_ADMIN_UPLOAD and username not in ADMIN_USERS:
        raise HTTPException(status_code=403, detail="❌ 只有管理員可以上傳文件")

    mime = magic.from_buffer(await file.read(2048), mime=True)
    await file.seek(0)

    if mime.startswith("image/") or mime.startswith("audio/"):
        raise HTTPException(status_code=400, detail="❌ 不允許上傳圖片或音頻檔案")

    allowed_types = [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/json",
        "text/x-python",
        "text/x-java",
        "application/javascript",
        "text/css",
        "application/x-php",
    ]
    if mime not in allowed_types:
        raise HTTPException(status_code=400, detail=f"❌ 不支援的文件類型: {mime}")

    save_path = os.path.join(UPLOAD_DOC_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ 存進 memory.db
    save_message_to_db(
        username,
        "system",
        f"📄 文件已上傳: {file.filename}",
        file_type="document",
        file_path=save_path
    )

    return {
        "msg": "✅ 文件上傳成功",
        "filename": file.filename,
        "mime": mime,
        "uploaded_by": username,
    }


# -----------------------------
# 上傳並問文件
# -----------------------------
@app.post("/upload/document/ask")
async def upload_and_ask_document(
    file: UploadFile = File(...),
    question: str = Form(...),
    username: str = Depends(get_current_username_from_db)   # ✅ 改用 token 拿 username
):
    # 1️⃣ 保存文件到硬碟
    save_path = os.path.join(UPLOAD_DOC_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ 存 DB - 上傳動作
    save_message_to_db(
        username,
        "system",
        f"📄 文件已上傳: {file.filename}",
        file_type="document",
        file_path=save_path
    )

    # 2️⃣ 抽取文件內容
    text_content = ""
    if file.filename.lower().endswith(".pdf"):
        with pdfplumber.open(save_path) as pdf:
            text_content = "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file.filename.lower().endswith(".docx"):
        doc = Document(save_path)
        text_content = "\n".join([para.text for para in doc.paragraphs])
    else:
        with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
            text_content = f.read()

    if not text_content.strip():
        raise HTTPException(status_code=400, detail="❌ 文件沒有可用的文字內容")

    # 3️⃣ 丟給 LLM
    prompt = f"""
這是使用者上傳的文件內容：

{text_content}

使用者問題：{question}

請根據文件內容來回答問題。
"""
    resp = ollama.chat(
        model="gpt-oss-20b",
        messages=[
            {"role": "system", "content": "你是一個助理，請幫助使用者理解文件內容。"},
            {"role": "user", "content": prompt}
        ],
        options={"num_predict": 500}
    )
    answer = resp["message"]["content"]

    # ✅ 存 DB - 使用者提問
    save_message_to_db(username, "user", question, file_type="document", file_path=save_path)

    # ✅ 存 DB - AI 回答
    save_message_to_db(username, "llm", answer)

    # 4️⃣ 回傳
    return {
        "filename": file.filename,
        "question": question,
        "answer": answer
    }


# -----------------------------
# 聊天紀錄 API
# -----------------------------
@app.get("/api/chat/recent")
def get_recent_history(limit: int = 20):
    conn = sqlite3.connect("memory.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            username TEXT,
            role TEXT,
            message TEXT,
            aitest TEXT,
            file_type TEXT,
            file_path TEXT,
            created_at TIMESTAMP,
            is_active BOOLEAN DEFAULT 1   -- 🆕 加入是否啟用 (1=顯示, 0=刪除/隱藏)
        )
    """)
    cur.execute("""
        SELECT id, username, role, message, created_at
        FROM messages
        WHERE is_active=1
        ORDER BY created_at DESC LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()

    history = [
        {"id": r[0], "username": r[1], "role": r[2], "text": r[3], "timestamp": r[4]}
        for r in reversed(rows)
    ]
    return {"history": history}

@app.delete("/auth/delete-picture")
def delete_user_picture(token: Optional[str] = Cookie(None)):
    """
    刪除目前使用者的頭像圖片與資料庫紀錄
    """
    if not token:
        raise HTTPException(status_code=401, detail="❌ 沒有登入 Token")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="❌ Token 無效")

        # 🔍 找使用者資料夾
        user_dir = os.path.join(USER_ROOT_DIR, "patient", str(user_id), "picture")
        if os.path.exists(user_dir):
            for file in os.listdir(user_dir):
                try:
                    os.remove(os.path.join(user_dir, file))
                except Exception as e:
                    print("刪除錯誤：", e)

        # 🔸 清除 DB 欄位
        conn = get_db()
        cur = conn.cursor()
        cur.execute("UPDATE patients SET user_picture=NULL WHERE id=?", (user_id,))
        conn.commit()
        conn.close()

        return {"msg": "✅ 頭像圖片已刪除"}

    except JWTError:
        raise HTTPException(status_code=401, detail="❌ Token 錯誤或過期")
    
@app.get("/api/patient/profile")
def get_patient_profile(current_user_id: str = Depends(get_current_user_id)):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            id, username, name, age, diagnosis, email, language,
            integral, verify, user_picture
        FROM patients WHERE id=?
    """, (current_user_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="User not found")


    # ✅ 確保前端拿到正確路徑（不重複 User）
    picture_path = row[9] or "static/default-user.png"
    picture_url = "/" + str(picture_path).lstrip("/")

    return {
        "user": {
            "id": row[0],
            "username": row[1],
            "name": row[2],
            "age": row[3],
            "diagnosis": row[4],
            "email": row[5],
            "language": row[6],
            "integral": row[7],
            "verify": row[8],
            "user_picture": picture_url  # 👈 新增這行，前端才能抓到
        }
    }

@app.get("/api/chat/history")
def get_all_history(limit: int = 50):
    conn = sqlite3.connect("memory.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT id, username, role, message, created_at
        FROM messages
        WHERE is_active=1
        ORDER BY created_at DESC LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()

    history = [
        {"id": r[0], "username": r[1], "role": r[2], "text": r[3], "timestamp": r[4]}
        for r in reversed(rows)
    ]
    return {"history": history}

@app.delete("/api/chat/history/{msg_id}")
def delete_one_message(msg_id: str):
    try:
        conn = sqlite3.connect("memory.db")
        cur = conn.cursor()
        cur.execute("UPDATE messages SET is_active=0 WHERE id=?", (msg_id,))
        conn.commit()
        conn.close()
        return {"msg": f"✅ 已隱藏訊息 {msg_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除訊息失敗: {str(e)}")
    
@app.post("/api/chat/hide-all")
def hide_all_messages():
    try:
        conn = sqlite3.connect("memory.db")  # ✅ correct database
        cur = conn.cursor()
        cur.execute("UPDATE messages SET is_active=0")  # ✅ correct table
        conn.commit()
        conn.close()
        return {"msg": "所有訊息已標記為隱藏"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失敗: {str(e)}")
    
@app.post("/auth/resend-verify")
def resend_verify_email(username: str):
    conn = get_db()
    cur = conn.cursor()

    # 允許傳 email 或 username 都可匹配
    cur.execute("""
        SELECT email FROM patients WHERE username=? OR email=?
    """, (username, username))
    row = cur.fetchone()

    if not row or not row[0]:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found or no email")

    email = row[0]
    verify_code = str(random.randint(100000, 999999))

    # 儲存此驗證碼
    cur.execute("""
        CREATE TABLE IF NOT EXISTS email_verifies (
            username TEXT PRIMARY KEY,
            code TEXT,
            created_at TIMESTAMP
        )
    """)
    cur.execute("REPLACE INTO email_verifies (username, code, created_at) VALUES (?, ?, ?)",
                (username, verify_code, datetime.now()))

    conn.commit()
    conn.close()

    # ✅ Gmail 寄信設定
    sender = "aaa2025819@gmail.com"
    smtp_host = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "aaa2025819@gmail.com"
    smtp_pass = "ssjuayuhpfugwltp"

    try:
        subject = "ASR Verification Code"
        body = f"""
Hello {username},

Your verification code is: {verify_code}

Please copy this code into the website to activate your account.

Thank you,
ASR+LLM System
"""
        import smtplib
        from email.mime.text import MIMEText
        from email.header import Header

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = sender
        msg["To"] = email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(sender, [email], msg.as_string())

        return {"msg": f"📩 驗證信已成功寄至 {email}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"寄信失敗: {str(e)}")
    
# ==========================================
# ✅ 忘記密碼：Send Verify Code + Reset Password
# ==========================================
class ForgotPasswordRequest(BaseModel):
    email: str

class VerifyResetCodeRequest(BaseModel):
    email: str
    code: str

class ResetPasswordRequest(BaseModel):
    email: str
    code: str
    new_password: str


@app.post("/auth/forgot-password/send-code")
def send_reset_code(data: ForgotPasswordRequest):
    """
    第一步：使用者輸入 Email → 寄出驗證碼
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT username FROM patients WHERE email=?", (data.email,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="查無此 Email，請確認輸入是否正確")

    username = row[0]
    reset_code = str(random.randint(100000, 999999))

    # 建立表格存放 reset code
    cur.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            email TEXT PRIMARY KEY,
            code TEXT,
            created_at TIMESTAMP
        )
    """)
    cur.execute("REPLACE INTO password_resets (email, code, created_at) VALUES (?, ?, ?)",
                (data.email, reset_code, datetime.now()))
    conn.commit()
    conn.close()

    # 寄信
    sender = "aaa2025819@gmail.com"
    smtp_host = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "aaa2025819@gmail.com"
    smtp_pass = "ssjuayuhpfugwltp"  # ⚠️ Gmail 應用專用密碼

    subject = "ASR Password Reset Code"
    body = f"""
Hello {username},

Your password reset code is: {reset_code}

Please enter this code on the website to reset your password.

Thank you,
ASR+LLM System
"""

    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = sender
        msg["To"] = data.email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(sender, [data.email], msg.as_string())

        return {"msg": f"📩 驗證碼已寄出至 {data.email}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"寄信失敗: {str(e)}")


@app.post("/auth/forgot-password/verify-code")
def verify_reset_code(data: VerifyResetCodeRequest):
    """
    第二步：使用者輸入驗證碼 → 驗證
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT code, created_at FROM password_resets WHERE email=?", (data.email,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="未找到驗證碼記錄")

    saved_code, created_at = row
    if data.code.strip() != saved_code:
        conn.close()
        raise HTTPException(status_code=400, detail="驗證碼錯誤")

    return {"msg": "✅ 驗證成功，請設定新密碼"}


import sys
from fastapi import HTTPException

import sys
from fastapi import HTTPException

@app.post("/auth/forgot-password/reset")
def reset_password(data: ResetPasswordRequest):
    conn = get_db()
    cur = conn.cursor()

    # 1) 檢查 reset code
    cur.execute("SELECT code FROM password_resets WHERE email=?", (data.email,))
    row = cur.fetchone()
    if not row or data.code.strip() != row[0].strip():
        conn.close()
        raise HTTPException(status_code=400, detail="驗證碼錯誤，請重新輸入")

    # 2) 更新密碼
    new_hash = hash_password(data.new_password)
    cur.execute(
        "UPDATE patients SET password_hash=? WHERE email=?",
        (new_hash, data.email),
    )
    conn.commit()

    # ★關鍵：看有沒有真的更新到資料
    print("DEBUG updated rows:", cur.rowcount); sys.stdout.flush()
    if cur.rowcount != 1:
        conn.close()
        raise HTTPException(status_code=404, detail="No patient row updated (email not found?)")

    # 3) 清除 reset code
    cur.execute("DELETE FROM password_resets WHERE email=?", (data.email,))
    conn.commit()
    conn.close()

    return {"msg": "✅ 密碼已重設成功，請重新登入"}

# --- 建立表格（安全保險，若表不存在自動建立一次） ---
def ensure_task_table():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        category TEXT CHECK(category IN ('daily','weekly','monthly')) NOT NULL,
        due_date TEXT,
        total_times INTEGER DEFAULT 0,
        progress_times INTEGER DEFAULT 0,
        integral INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()

def ensure_task_table_v2():
    """
    ✅ tasks table v2
    - 支援 user 多 task
    - 支援 training task: super_type/sub_type/language/day_index/allocated_at
    - 支援 sentence/vocab: task_kind
    - 預留 task_key
    """
    conn = get_db()
    cur = conn.cursor()

    # 1) 先確保舊表存在（沿用你原本結構）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        category TEXT CHECK(category IN ('daily','weekly','monthly')) NOT NULL,
        due_date TEXT,
        total_times INTEGER DEFAULT 0,
        progress_times INTEGER DEFAULT 0,
        integral INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()

    # 2) 增加欄位（SQLite 允許逐個 ADD COLUMN）
    cur.execute("PRAGMA table_info(tasks);")
    cols = {r[1] for r in cur.fetchall()}

    def add_col(sql: str):
        cur.execute(sql)
        conn.commit()

    # ✅ reward claim (one-time +50 from Task page)
    if "reward_claimed" not in cols:
        add_col("ALTER TABLE tasks ADD COLUMN reward_claimed INTEGER NOT NULL DEFAULT 0;")

    if "reward_claimed_at" not in cols:
        add_col("ALTER TABLE tasks ADD COLUMN reward_claimed_at TIMESTAMP;")

    if "user_id" not in cols:
        add_col("ALTER TABLE tasks ADD COLUMN user_id TEXT;")  # 舊資料可 NULL

    if "super_type" not in cols:
        add_col("ALTER TABLE tasks ADD COLUMN super_type TEXT;")

    if "sub_type" not in cols:
        add_col("ALTER TABLE tasks ADD COLUMN sub_type TEXT;")

    if "language" not in cols:
        add_col("ALTER TABLE tasks ADD COLUMN language TEXT NOT NULL DEFAULT 'en';")

    if "allocated_at" not in cols:
        add_col("ALTER TABLE tasks ADD COLUMN allocated_at TIMESTAMP;")

    if "valid_days" not in cols:
        add_col("ALTER TABLE tasks ADD COLUMN valid_days INTEGER NOT NULL DEFAULT 7;")

    if "day_index" not in cols:
        add_col("ALTER TABLE tasks ADD COLUMN day_index INTEGER NOT NULL DEFAULT 1;")

    if "is_active" not in cols:
        add_col("ALTER TABLE tasks ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1;")

    # 你要的：sentence/vocab
    if "task_kind" not in cols:
        add_col("ALTER TABLE tasks ADD COLUMN task_kind TEXT NOT NULL DEFAULT 'vocab';")

    # 預留：同日同 subtype 多 task 時用
    if "task_key" not in cols:
        add_col("ALTER TABLE tasks ADD COLUMN task_key TEXT;")

    if "updated_at" not in cols:
        add_col("ALTER TABLE tasks ADD COLUMN updated_at TIMESTAMP;")
        # 先把現有 row 補值
        cur.execute("UPDATE tasks SET updated_at = COALESCE(updated_at, CURRENT_TIMESTAMP)")
    # 3) indexes（避免查詢慢）
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_tasks_user_training
        ON tasks(user_id, super_type, sub_type, language, is_active, allocated_at, day_index);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_tasks_user_created
        ON tasks(user_id, created_at);
    """)

    conn.commit()
    conn.close()


def _task_allowed_day_index(allocated_at: str | None) -> int | None:
    """
    allocated_at: DB 字串（YYYY-MM-DD HH:MM:SS）或 None
    回傳 allowed_day_index = days_since + 1 (1..7)
    超過 7 日回傳 None
    """
    if not allocated_at:
        return None
    try:
        base_date = datetime.strptime(str(allocated_at)[:19], "%Y-%m-%d %H:%M:%S").date()
    except Exception:
        # 有些 sqlite timestamp 可能是 ISO
        try:
            base_date = datetime.fromisoformat(str(allocated_at)).date()
        except Exception:
            return None

    today = datetime.now().date()
    days_since = (today - base_date).days
    if days_since < 0 or days_since > 6:
        return None
    return days_since + 1

class TodayTrainingTaskResp(BaseModel):
    available: bool
    allowed_day_index: Optional[int] = None
    reason: Optional[str] = None
    task: Optional[dict] = None

@app.get("/api/tasks/training/today", response_model=TodayTrainingTaskResp)
def api_tasks_training_today(
    super_type: str = Query(...),
    sub_type: str = Query(...),
    language: str = Query("cn"),
    task_kind: str = Query("vocab"),
    current_user_id: str = Depends(get_current_user_id),
):
    lang = normalize_practice_language(language)
    super_type = (super_type or "").strip()
    sub_type = (sub_type or "").strip()
    task_kind = (task_kind or "vocab").strip().lower()

    if task_kind not in ("vocab", "sentence"):
        raise HTTPException(status_code=400, detail="Invalid task_kind (vocab|sentence)")

    conn = get_db()
    cur = conn.cursor()
    try:
        # 1) 找最新一批 allocated_at（batch）
        cur.execute("""
            SELECT allocated_at
            FROM tasks
            WHERE user_id=?
              AND super_type=?
              AND sub_type=?
              AND language=?
              AND task_kind=?
              AND is_active=1
              AND allocated_at IS NOT NULL
            ORDER BY datetime(allocated_at) DESC
            LIMIT 1
        """, (current_user_id, super_type, sub_type, lang, task_kind))
        r = cur.fetchone()
        if not r or not r[0]:
            return TodayTrainingTaskResp(available=False, reason="NO_TASK")

        allocated_at = r[0]

        # 2) 算今日 allowed day (1..7)，過期就 EXPIRED
        allowed = _task_allowed_day_index(allocated_at)
        if allowed is None:
            return TodayTrainingTaskResp(available=False, reason="EXPIRED", allowed_day_index=None)

        # 3) 直接攞「今日嗰一日」(day_index == allowed) 嗰筆 task
        cur.execute("""
            SELECT
              id, user_id, super_type, sub_type, task_kind, language,
              allocated_at, day_index, total_times, progress_times, status, is_active
            FROM tasks
            WHERE user_id=?
              AND super_type=?
              AND sub_type=?
              AND language=?
              AND task_kind=?
              AND is_active=1
              AND datetime(allocated_at)=datetime(?)
              AND day_index=?
            LIMIT 1
        """, (current_user_id, super_type, sub_type, lang, task_kind, allocated_at, int(allowed)))
        row = cur.fetchone()

        # 今日應做嗰筆都搵唔到：就唔提供
        if not row:
            return TodayTrainingTaskResp(
                available=False,
                reason="NOT_TODAY_TASK_DAY",
                allowed_day_index=allowed
            )

        (tid, uid, st, sb, kind, lg, alloc, day_index,
         total_times, progress_times, status, is_active) = row

        task_obj = {
            "id": tid,
            "user_id": uid,
            "super_type": st,
            "sub_type": sb,
            "type": kind,  # 你前端用 taskInfo.type
            "language": lg,
            "allocated_at": alloc,
            "day_index": int(day_index),
            "total_times": int(total_times or 0),
            "progress_times": int(progress_times or 0),
            "status": status,
            "is_active": int(is_active or 0),
        }

        return TodayTrainingTaskResp(
            available=True,
            allowed_day_index=allowed,
            reason=None,
            task=task_obj
        )

    finally:
        conn.close()

class TrainingTaskIncrementReq(BaseModel):
    task_id: str
    increment: int = 1

class TrainingTaskIncrementResp(BaseModel):
    msg: str
    task_id: str
    progress_times: int
    total_times: int
    status: str

@app.post("/api/tasks/training/increment", response_model=TrainingTaskIncrementResp)
def api_tasks_training_increment(
    data: TrainingTaskIncrementReq,
    current_user_id: str = Depends(get_current_user_id),
):
    inc = int(data.increment or 1)
    if inc <= 0 or inc > 10:
        raise HTTPException(status_code=400, detail="Invalid increment")

    conn = get_db()
    cur = conn.cursor()

    # 1) load task
    cur.execute("""
        SELECT
          id, user_id, super_type, sub_type, task_kind, language,
          allocated_at, day_index, total_times, progress_times, status, is_active
        FROM tasks
        WHERE id=?
        LIMIT 1
    """, (data.task_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")

    (tid, uid, super_type, sub_type, task_kind, lang,
     allocated_at, day_index, total_times, progress_times, status, is_active) = row

    if uid != current_user_id:
        conn.close()
        raise HTTPException(status_code=403, detail="Not allowed")

    if int(is_active or 0) != 1:
        conn.close()
        raise HTTPException(status_code=400, detail="Task inactive")

    allowed = _task_allowed_day_index(allocated_at)
    if allowed is None:
        conn.close()
        raise HTTPException(status_code=400, detail="Task expired")

    if int(day_index) != int(allowed):
        conn.close()
        raise HTTPException(status_code=400, detail=f"Not allowed day. allowed_day_index={allowed}")

    # 2) increment
    total = int(total_times or 0)
    prog = int(progress_times or 0)
    new_prog = prog + inc
    if total > 0:
        new_prog = min(new_prog, total)

    was_completed = (str(status or "").lower() == "completed")

    new_status = status
    if total > 0 and new_prog >= total:
        new_status = "completed"

    cur.execute("""
        UPDATE tasks
        SET progress_times=?,
            status=?
        WHERE id=?
    """, (new_prog, new_status, tid))

    # ✅ 若這一日變 completed，嘗試把整批 day1..7 做完就關閉（is_active=0）
    if str(new_status or "").lower() == "completed":
        deactivate_batch_if_fully_completed(
            cur,
            user_id=current_user_id,
            super_type=super_type,
            sub_type=sub_type,
            language=lang,        # 這裡是 tasks.language 值（cn/en）
            task_kind=task_kind,
            allocated_at=allocated_at,
        )

    became_completed_now = (not was_completed) and (str(new_status).lower() == "completed")
    if became_completed_now:
        award_points_on_task_completed(cur, current_user_id, tid, points=100)

    conn.commit()
    conn.close()

    return TrainingTaskIncrementResp(
        msg="ok",
        task_id=tid,
        progress_times=new_prog,
        total_times=total,
        status=new_status
    )
# ------------------------------------------------------------
# 📌 1️⃣ 建立任務（管理員或外部頁面呼叫）
# ------------------------------------------------------------
@app.post("/api/tasks/create")
def create_task(
    title: str = Form(...),
    description: str = Form(""),
    category: str = Form(...),
    due_date: str = Form(None),
    total_times: int = Form(0),
    integral: int = Form(0),
):
    conn = get_db()
    cur = conn.cursor()
    task_id = str(uuid.uuid4())

    try:
        cur.execute("""
            INSERT INTO tasks (id, title, description, category, due_date, total_times, progress_times, integral)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
        """, (task_id, title, description, category, due_date, total_times, integral))
        conn.commit()
        return {"msg": "✅ 任務建立成功", "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"建立任務失敗：{str(e)}")
    finally:
        conn.close()


# ------------------------------------------------------------
# 📌 2️⃣ 使用者領取任務（將 task_id 存入 patients）
# ------------------------------------------------------------
@app.post("/api/tasks/take")
def take_task(
    task_id: str = Form(...),
    current_user_id: str = Depends(get_current_user_id)
):
    conn = get_db()
    cur = conn.cursor()

    # 確認任務存在
    cur.execute("SELECT id FROM tasks WHERE id=?", (task_id,))
    task = cur.fetchone()
    if not task:
        conn.close()
        raise HTTPException(status_code=404, detail="任務不存在")

    # 更新用戶 task_id
    cur.execute("UPDATE patients SET task_id=? WHERE id=?", (task_id, current_user_id))
    conn.commit()
    conn.close()
    return {"msg": f"✅ 已領取任務 {task_id}"}


# ------------------------------------------------------------
# 📌 3️⃣ 更新任務進度與積分（使用者完成一次任務）
# ------------------------------------------------------------
@app.post("/api/tasks/set-progress")
def set_task_progress(
    increment: int = Form(1),
    current_user_id: str = Depends(get_current_user_id)
):
    """
    當使用者完成一次任務時呼叫此API：
    - 進度 + increment
    - 若完成全部次數 → 狀態改為 completed
    - 並增加使用者積分
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        # 查出該使用者的任務 ID
        cur.execute("SELECT task_id FROM patients WHERE id=?", (current_user_id,))
        row = cur.fetchone()
        if not row or not row[0]:
            raise HTTPException(status_code=400, detail="此使用者目前沒有任務")

        task_id = row[0]

        # 查任務資料
        cur.execute("SELECT total_times, progress_times, integral FROM tasks WHERE id=?", (task_id,))
        trow = cur.fetchone()
        if not trow:
            raise HTTPException(status_code=404, detail="找不到任務")

        total, progress, task_integral = trow
        new_progress = min(progress + increment, total)
        status = "completed" if new_progress >= total and total > 0 else "pending"

        # 更新任務狀態與進度
        cur.execute(
            "UPDATE tasks SET progress_times=?, status=? WHERE id=?",
            (new_progress, status, task_id)
        )

        # 若達成 -> 增加積分給該使用者
        if status == "completed":
            cur.execute(
                "UPDATE patients SET integral = integral + ? WHERE id=?",
                (task_integral, current_user_id)
            )

        conn.commit()
        return {
            "msg": f"✅ 任務進度更新成功 ({new_progress}/{total})",
            "status": status,
            "added_integral": task_integral if status == "completed" else 0
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新任務進度發生錯誤：{str(e)}")
    finally:
        conn.close()


# ------------------------------------------------------------
# 📌 4️⃣ 瀏覽任務清單（分類 daily/weekly/monthly）
# ------------------------------------------------------------
@app.get("/api/tasks/list")
def list_tasks(
    category: str = Query(None),
    current_user_id: str = Depends(get_current_user_id)
):
    conn = get_db()
    cur = conn.cursor()
    sql = "SELECT id, title, description, category, due_date, total_times, progress_times, integral, status FROM tasks"
    params = []
    if category:
        sql += " WHERE category=?"
        params.append(category)
    sql += " ORDER BY created_at DESC"
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    result = [
        {
            "id": r[0],
            "title": r[1],
            "description": r[2],
            "category": r[3],
            "due_date": r[4],
            "total_times": r[5],
            "progress_times": r[6],
            "integral": r[7],
            "status": r[8],
        }
        for r in rows
    ]
    return {"tasks": result}

def ensure_staff_verify_table():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS staff_email_verifies (
        email TEXT,
        role TEXT,
        code TEXT,
        created_at TIMESTAMP,
        PRIMARY KEY (email, role)
    )
    """)
    conn.commit()
    conn.close()
# ==============================
# ✅ Staff (SLP / Admin) Auth APIs
# ==============================

class StaffRegister(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    name: Optional[str] = None
    role: str  # 'slp' or 'admin'
    license_number: Optional[str] = None
    organization: Optional[str] = None
    description: Optional[str] = None   # ✅ 新增
    language: Optional[str] = "en"

class StaffLogin(BaseModel):
    email: str
    password: str
    role: str   # 'slp' or 'admin'

def get_current_staff_id(token: str = Depends(oauth2_scheme)):
    """
    staffToken 使用同一個 oauth2_scheme，但 payload 會帶 staff=1
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        staff_id = payload.get("sub")
        is_staff = payload.get("staff")
        if not staff_id or is_staff != 1:
            raise HTTPException(status_code=401, detail="Invalid staff token")
        return staff_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")


@app.post("/staff/register")
def staff_register(data: StaffRegister):
    ensure_staff_verify_table()
    """
    SLP/Admin 申請帳號：
    - 建立 staff_users (verify=0, is_approved=0)
    - 寄出 email 驗證碼，寫入 staff_email_verifies
    - ✅ 只有 verify=1 才算完成申請（後續才允許登入 / 才允許進審核流程）
    """
    role = (data.role or "").strip()
    if role not in ("slp", "admin"):
        raise HTTPException(status_code=400, detail="Invalid role (must be 'slp' or 'admin')")

    if not data.email:
        raise HTTPException(status_code=400, detail="Email is required")

    if role == "slp" and (not data.license_number or not data.license_number.strip()):
        raise HTTPException(status_code=400, detail="License number is required for SLP")

    conn = get_db()
    cur = conn.cursor()

    # username unique
    cur.execute("SELECT 1 FROM staff_users WHERE username=?", (data.username,))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="❌ Username already exists")

    # ✅ email+role unique（符合你要的：同 role 不可重複，不同 role 可重複）
    cur.execute("SELECT 1 FROM staff_users WHERE email=? AND role=?", (data.email, role))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="❌ Email already exists for this role")

    # 產生 6 位驗證碼
    verify_code = str(random.randint(100000, 999999))

    # 寄送驗證信（沿用你 patient 的寄信設定）
    try:
        sender = "aaa2025819@gmail.com"
        smtp_host = "smtp.gmail.com"
        smtp_port = 587
        smtp_user = "aaa2025819@gmail.com"
        smtp_pass = "ssjuayuhpfugwltp"

        subject = "Staff Email Verification Code"
        body = f"""
Hello {data.username},

Your staff verification code is: {verify_code}

Role: {role}

Please enter this code to verify your staff email before your application can be submitted.

Thank you,
ASR+LLM System
"""
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = sender
        msg["To"] = data.email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(sender, [data.email], msg.as_string())

    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Email send failed: {e}")

    # 建立 staff_users（verify=0）
    staff_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO staff_users
        (id, username, password_hash, name, email, role, license_number, organization, description, language,
        verify, staff_picture, is_approved)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0)
    """, (
        staff_id,
        data.username,
        hash_password(data.password),
        data.name,
        data.email,
        role,
        data.license_number,
        data.organization,
        data.description,      # ✅ 新增
        data.language
    ))

    # ✅ 儲存驗證碼（用 email+role 做 key 才不會撞）
    # 你的 staff_email_verifies 目前只有 email PK，會撞不同 role。
    # 👉 建議你把 staff_email_verifies 改成 (email, role) 複合 PK。
    # 暫時先用 email+role 拼成一個 key 字串來避開撞（不改 DB 的情況下）。
    cur.execute(
        "INSERT OR REPLACE INTO staff_email_verifies (email, role, code, created_at) VALUES (?, ?, ?, ?)",
        (data.email, role, verify_code, datetime.now())
    )

    conn.commit()
    conn.close()

    return {
        "msg": f"📩 Staff verification code sent to {data.email}. Please verify before your application is submitted.",
        "id": staff_id,
        "username": data.username,
        "email": data.email,
        "role": role,
        "verify": 0,
        "is_approved": 0
    }

@app.post("/staff/login")
def staff_login(
    data: StaffLogin,
    response: Response,
    send_cookie: bool = Query(SEND_USERNAME_COOKIE_DEFAULT, description="是否下發 staff_email cookie"),
):
    print("DEBUG staff_login role =", repr(data.role))
    role = (data.role or "").strip()
    if role not in ("slp", "admin"):
        raise HTTPException(status_code=400, detail="Invalid role")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, password_hash, role, is_approved, username, email, verify
        FROM staff_users
        WHERE email=? AND role=?
    """, (data.email, role))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    staff_id, password_hash_db, role, is_approved, username, email, verify = row

    if not verify_password(data.password, password_hash_db):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # ✅ 必須 email verify=1 才能登入（符合你的規則）
    if int(verify) != 1:
        raise HTTPException(status_code=403, detail="Email not verified")

    # approval gate
    if int(is_approved) == 0:
        raise HTTPException(status_code=403, detail="Account pending approval")
    if int(is_approved) == 2:
        raise HTTPException(status_code=403, detail="Account rejected")

    expire = datetime.utcnow() + timedelta(hours=12)
    token_data = {"sub": staff_id, "exp": expire, "staff": 1, "role": role}
    staff_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

    if send_cookie and email:
        response.set_cookie(
            key="staff_email",
            value=email,
            httponly=False,
            samesite="Lax",
        )

    return {
        "msg": "Staff login successful",
        "staffToken": staff_token,
        "id": staff_id,
        "username": username,
        "email": email,
        "role": role
    }

@app.get("/staff/me")
def staff_me(current_staff_id: str = Depends(get_current_staff_id)):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, username, name, email, role, license_number, organization,
               description, language, verify, is_approved, created_at, staff_picture
        FROM staff_users
        WHERE id=?
    """, (current_staff_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Staff not found")

    return {
        "staff": {
            "id": row[0],
            "username": row[1],
            "name": row[2],
            "email": row[3],
            "role": row[4],
            "license_number": row[5],
            "organization": row[6],
            "description": row[7],
            "language": row[8],
            "verify": row[9],
            "is_approved": row[10],
            "created_at": row[11],
            "staff_picture": row[12],
        }
    }
class StaffVerifyCodeRequest(BaseModel):
    email: str
    role: str
    code: str

@app.post("/staff/verify-code")
def staff_verify_code(data: StaffVerifyCodeRequest):
    """
    staff email 驗證：
    - 比對 staff_email_verifies 的 code（用 email + role）
    - 成功後把 staff_users.verify 設為 1
    - 刪除驗證碼
    """
    role = (data.role or "").strip()
    if role not in ("slp", "admin"):
        raise HTTPException(status_code=400, detail="Invalid role")

    email = (data.email or "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    conn = get_db()
    cur = conn.cursor()
    try:
        # ✅ 用 (email, role) 查驗證碼
        cur.execute(
            "SELECT code FROM staff_email_verifies WHERE email=? AND role=?",
            (email, role)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Verification code not found")

        db_code = row[0]
        if str(data.code).strip() != str(db_code).strip():
            raise HTTPException(status_code=400, detail="Incorrect verification code")

        # ✅ 更新 staff_users.verify=1
        cur.execute(
            "UPDATE staff_users SET verify=1 WHERE email=? AND role=?",
            (email, role)
        )
        conn.commit()

        # ✅ 刪除驗證碼（同樣用 email + role）
        cur.execute(
            "DELETE FROM staff_email_verifies WHERE email=? AND role=?",
            (email, role)
        )
        conn.commit()

        return {"msg": "✅ Staff email verified. Your application is now submitted."}

    finally:
        conn.close()

@app.post("/staff/upload-picture")
async def upload_staff_picture(
    file: UploadFile = File(...),
    staff_token: Optional[str] = Cookie(None),
    token: Optional[str] = Cookie(None),   # 保留兼容舊的
    response: Response = None
):
    jwt_token = staff_token or token
    if not jwt_token:
        raise HTTPException(status_code=401, detail="❌ 沒有登入 Token")

    # ✅ 正確：用 jwt_token decode
    try:
        payload = jwt.decode(jwt_token, SECRET_KEY, algorithms=[ALGORITHM])
        staff_id = payload.get("sub")
        is_staff = payload.get("staff")
        if not staff_id or is_staff != 1:
            raise HTTPException(status_code=401, detail="❌ Token 不是 staff")
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"❌ Token 錯誤或過期: {str(e)}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="❌ 未收到檔案或檔名為空")

    filename = file.filename.lower()
    if not (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
        raise HTTPException(status_code=400, detail="❌ 僅允許 PNG / JPG / JPEG")

    contents = await file.read()
    file_size_kb = len(contents) / 1024
    if file_size_kb > 1000:
        raise HTTPException(status_code=400, detail="❌ 檔案大小超過限制（最大 1MB）")

    # 存檔路徑
    staff_dir = os.path.join(USER_ROOT_DIR, "staff", str(staff_id), "picture")
    os.makedirs(staff_dir, exist_ok=True)

    # 刪舊圖
    for old in os.listdir(staff_dir):
        try:
            os.remove(os.path.join(staff_dir, old))
        except Exception:
            pass

    ext = ".png" if filename.endswith(".png") else ".jpg"
    file_name = f"profile{ext}"
    save_path = os.path.join(staff_dir, file_name)
    with open(save_path, "wb") as f:
        f.write(contents)

    # 更新 DB
    relative_path = f"User/staff/{staff_id}/picture/{file_name}"
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE staff_users SET staff_picture=? WHERE id=?", (relative_path, staff_id))
    conn.commit()
    conn.close()

    public_url = f"/{relative_path}"

    # 可選寫 cookie
    if response:
        response.set_cookie(
            key="staff_picture_url",
            value=public_url,
            httponly=False,
            samesite="Lax",
            secure=True
        )

    return {"msg": "✅ 上傳成功", "staff_picture_url": public_url}


class StaffResendVerifyRequest(BaseModel):
    email: str
    role: str

@app.post("/staff/resend-verify")
def staff_resend_verify(data: StaffResendVerifyRequest):
    ensure_staff_verify_table()
    role = (data.role or "").strip()
    if role not in ("slp", "admin"):
        raise HTTPException(status_code=400, detail="Invalid role")

    email = (data.email or "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    conn = get_db()
    cur = conn.cursor()

    # 確認 staff 帳號存在（email+role）
    cur.execute("SELECT username FROM staff_users WHERE email=? AND role=?", (email, role))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Staff account not found")

    username = row[0]
    verify_code = str(random.randint(100000, 999999))

    # 更新/寫入驗證碼
    cur.execute(
        "INSERT OR REPLACE INTO staff_email_verifies (email, role, code, created_at) VALUES (?, ?, ?, ?)",
        (email, role, verify_code, datetime.now())
    )
    conn.commit()
    conn.close()

    # 寄信（沿用你既有 Gmail 設定）
    try:
        sender = "aaa2025819@gmail.com"
        smtp_host = "smtp.gmail.com"
        smtp_port = 587
        smtp_user = "aaa2025819@gmail.com"
        smtp_pass = "ssjuayuhpfugwltp"

        subject = "Staff Email Verification Code"
        body = f"""
Hello {username},

Your staff verification code is: {verify_code}

Role: {role}

Thank you,
ASR+LLM System
"""
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = sender
        msg["To"] = email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(sender, [email], msg.as_string())

        return {"msg": f"📩 Staff verification code resent to {email}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email send failed: {str(e)}")


# ==========================================
# ✅ Staff Forgot Password (SLP/Admin)
# ==========================================
class StaffForgotPasswordRequest(BaseModel):
    email: str
    role: str  # slp | admin

class StaffVerifyResetCodeRequest(BaseModel):
    email: str
    role: str
    code: str

class StaffResetPasswordRequest(BaseModel):
    email: str
    role: str
    code: str
    new_password: str

def ensure_staff_password_reset_table():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS staff_password_resets (
            email TEXT NOT NULL,
            role TEXT NOT NULL,
            code TEXT NOT NULL,
            created_at TIMESTAMP,
            PRIMARY KEY(email, role)
        )
    """)
    conn.commit()
    conn.close()


@app.post("/staff/forgot-password/send-code")
def staff_send_reset_code(data: StaffForgotPasswordRequest):
    role = (data.role or "").strip()
    if role not in ("slp", "admin"):
        raise HTTPException(status_code=400, detail="Invalid role")

    email = (data.email or "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT username FROM staff_users WHERE email=? AND role=?", (email, role))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Staff account not found")

    username = row[0]
    reset_code = str(random.randint(100000, 999999))

    ensure_staff_password_reset_table()
    cur.execute(
        "INSERT OR REPLACE INTO staff_password_resets (email, role, code, created_at) VALUES (?, ?, ?, ?)",
        (email, role, reset_code, datetime.now())
    )
    conn.commit()
    conn.close()

    # send email (reuse your Gmail settings)
    sender = "aaa2025819@gmail.com"
    smtp_host = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "aaa2025819@gmail.com"
    smtp_pass = "ssjuayuhpfugwltp"

    subject = "ASR Staff Password Reset Code"
    body = f"""
Hello {username},

Your staff password reset code is: {reset_code}

Role: {role}

If you did not request this, please ignore this email.
"""

    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = sender
        msg["To"] = email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(sender, [email], msg.as_string())

        return {"msg": f"📩 Reset code sent to {email}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email send failed: {str(e)}")


@app.post("/staff/forgot-password/verify-code")
def staff_verify_reset_code(data: StaffVerifyResetCodeRequest):
    role = (data.role or "").strip()
    if role not in ("slp", "admin"):
        raise HTTPException(status_code=400, detail="Invalid role")

    email = (data.email or "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    ensure_staff_password_reset_table()

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT code, created_at FROM staff_password_resets WHERE email=? AND role=?",
        (email, role)
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Reset code not found")

    saved_code = row[0]
    if str(data.code).strip() != str(saved_code).strip():
        raise HTTPException(status_code=400, detail="Incorrect reset code")

    return {"msg": "✅ Code verified. Please set a new password."}


@app.post("/staff/forgot-password/reset")
def staff_reset_password(data: StaffResetPasswordRequest):
    role = (data.role or "").strip()
    if role not in ("slp", "admin"):
        raise HTTPException(status_code=400, detail="Invalid role")

    email = (data.email or "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    ensure_staff_password_reset_table()

    conn = get_db()
    cur = conn.cursor()

    # 1) verify code
    cur.execute(
        "SELECT code FROM staff_password_resets WHERE email=? AND role=?",
        (email, role)
    )
    row = cur.fetchone()
    if not row or str(data.code).strip() != str(row[0]).strip():
        conn.close()
        raise HTTPException(status_code=400, detail="Incorrect reset code")

    # 2) update password
    new_hash = hash_password(data.new_password)
    cur.execute(
        "UPDATE staff_users SET password_hash=? WHERE email=? AND role=?",
        (new_hash, email, role)
    )
    conn.commit()

    if cur.rowcount != 1:
        conn.close()
        raise HTTPException(status_code=404, detail="No staff row updated")

    # 3) clear reset code
    cur.execute(
        "DELETE FROM staff_password_resets WHERE email=? AND role=?",
        (email, role)
    )
    conn.commit()
    conn.close()

    return {"msg": "✅ Password reset successful. Please login again."}


class CreateServiceSessionReq(BaseModel):
    topic: Optional[str] = None
    first_message: str

@app.post("/api/service/session")
def api_create_service_session(
    data: CreateServiceSessionReq,
    current_user_id: str = Depends(get_current_user_id)
):

    session_id = str(uuid.uuid4())
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO service_sessions (id, patient_id, slp_id, topic, priority, status)
            VALUES (?, ?, NULL, ?, 0, 'queued')
        """, (session_id, current_user_id, data.topic))

        # 第一則訊息
        msg_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO service_messages (id, session_id, sender_type, sender_id, message, message_type)
            VALUES (?, ?, 'patient', ?, ?, 'text')
        """, (msg_id, session_id, current_user_id, data.first_message))

        cur.execute("""
            UPDATE service_sessions
            SET last_message_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (session_id,))

        conn.commit()
    finally:
        conn.close()

    audit_log(
        "patient",
        current_user_id,
        "CREATE_SERVICE_SESSION",
        "service_session",
        session_id,
        metadata=json.dumps({"topic": data.topic}, ensure_ascii=False),
    )

    # ✅ 建立後嘗試 auto allocate（規則：只有 waiting SLP 才會派）
    system_auto_allocate_one_db()

    return {"session_id": session_id}


@app.get("/api/service/session/{session_id}")

def api_get_service_session(
    session_id: str,
    current_user_id: str = Depends(get_current_user_id)
):

    conn = get_db()
    cur = conn.cursor()
    refresh_one_session_priority(cur, session_id)
    conn.commit()
    cur.execute("""
        SELECT id, patient_id, slp_id, topic, priority, status, created_at, last_message_at, closed_at, closed_by
        FROM service_sessions
        WHERE id=?
    """, (session_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="session not found")

    # ✅ 只有該病患本人可以看（SLP 版另外做）
    if row[1] != current_user_id:
        raise HTTPException(status_code=403, detail="not allowed")

    return {
        "id": row[0],
        "patient_id": row[1],
        "slp_id": row[2],
        "topic": row[3],
        "priority": row[4],
        "status": row[5],
        "created_at": row[6],
        "last_message_at": row[7],
        "closed_at": row[8],
        "closed_by": row[9],
    }


@app.get("/api/service/session/{session_id}/messages")
def api_get_service_messages(
    session_id: str,
    current_user_id: str = Depends(get_current_user_id)
):
    ensure_customer_service_tables()

    conn = get_db()
    cur = conn.cursor()

    # 權限：病患本人
    cur.execute("SELECT patient_id FROM service_sessions WHERE id=?", (session_id,))
    srow = cur.fetchone()
    if not srow:
        conn.close()
        raise HTTPException(status_code=404, detail="session not found")
    if srow[0] != current_user_id:
        conn.close()
        raise HTTPException(status_code=403, detail="not allowed")

    cur.execute("""
        SELECT id, sender_type, sender_id, message, message_type, created_at
        FROM service_messages
        WHERE session_id=?
        ORDER BY created_at ASC
    """, (session_id,))
    items = [
        {
            "id": r[0],
            "sender_type": r[1],
            "sender_id": r[2],
            "message": r[3],
            "message_type": r[4],
            "created_at": r[5],
        }
        for r in cur.fetchall()
    ]
    conn.close()
    return {"items": items}

class PostServiceMessageReq(BaseModel):
    message: str

@app.post("/api/service/session/{session_id}/messages")
def api_post_service_message(
    session_id: str,
    data: PostServiceMessageReq,
    current_user_id: str = Depends(get_current_user_id)
):
    ensure_customer_service_tables()

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT patient_id, status FROM service_sessions WHERE id=?", (session_id,))
        srow = cur.fetchone()
        if not srow:
            raise HTTPException(status_code=404, detail="session not found")
        if srow[0] != current_user_id:
            raise HTTPException(status_code=403, detail="not allowed")
        if srow[1] in ("closed", "cancelled"):
            raise HTTPException(status_code=400, detail="session already closed")

        msg_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO service_messages (id, session_id, sender_type, sender_id, message, message_type)
            VALUES (?, ?, 'patient', ?, ?, 'text')
        """, (msg_id, session_id, current_user_id, data.message))

        cur.execute("UPDATE service_sessions SET last_message_at=CURRENT_TIMESTAMP WHERE id=?", (session_id,))
        conn.commit()

        audit_log("patient", current_user_id, "POST_SERVICE_MESSAGE", "service_message", msg_id,
                  metadata=json.dumps({"session_id": session_id}, ensure_ascii=False))
        return {"message_id": msg_id}
    finally:
        conn.close()

@app.post("/api/service/session/{session_id}/close")
def api_close_service_session_customer(
    session_id: str,
    current_user_id: str = Depends(get_current_user_id)
):
    ensure_customer_service_tables()

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT patient_id, status FROM service_sessions WHERE id=?", (session_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="session not found")
        if row[0] != current_user_id:
            raise HTTPException(status_code=403, detail="not allowed")
        if row[1] == "closed":
            return {"msg": "already closed"}

        cur.execute("""
            UPDATE service_sessions
            SET status='closed', closed_at=CURRENT_TIMESTAMP, closed_by='patient'
            WHERE id=?
        """, (session_id,))
        conn.commit()

        audit_log("patient", current_user_id, "CLOSE_SERVICE_SESSION", "service_session", session_id)
        return {"msg": "closed"}
    finally:
        conn.close()


def get_current_staff_payload(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        staff_id = payload.get("sub")
        is_staff = payload.get("staff")
        if not staff_id or is_staff != 1:
            raise HTTPException(status_code=401, detail="Invalid staff token")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")
    
def require_admin(payload: dict):
    if not payload or payload.get("staff") != 1 or payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    

class SLPOnlineReq(BaseModel):
    online: bool

@app.post("/api/slp/online")
def api_slp_set_online(
    data: SLPOnlineReq,
    payload: dict = Depends(get_current_staff_payload)
):
    ensure_customer_service_tables()

    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP can set online")

    slp_id = payload["sub"]
    slp_set_online_db(slp_id, bool(data.online))
    audit_log("slp", slp_id, "SLP_SET_ONLINE", "slp_presence", slp_id,
              metadata=json.dumps({"online": bool(data.online)}, ensure_ascii=False))
    return {"slp_id": slp_id, "online": bool(data.online)}

class SLPWaitReq(BaseModel):
    waiting: bool  # True=開始等待；False=取消等待

@app.post("/api/slp/wait")
def api_slp_wait(
    data: SLPWaitReq,
    payload: dict = Depends(get_current_staff_payload)
):
    ensure_customer_service_tables()

    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP can wait")

    slp_id = payload["sub"]
    slp_wait_db(slp_id, bool(data.waiting))

    audit_log("slp", slp_id, "SLP_SET_WAITING", "slp_presence", slp_id,
              metadata=json.dumps({"waiting": bool(data.waiting)}, ensure_ascii=False))

    # ✅ 開始等待後，讓系統嘗試派一單（如果有 queued）
    if data.waiting:
        system_auto_allocate_one_db()

    return {"slp_id": slp_id, "waiting": bool(data.waiting)}

@app.post("/api/slp/heartbeat")
def api_slp_heartbeat(payload: dict = Depends(get_current_staff_payload)):
    ensure_customer_service_tables()

    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    slp_id = payload["sub"]
    slp_heartbeat_db(slp_id)
    return {"slp_id": slp_id, "ok": True}

@app.get("/api/slp/sessions")
def api_slp_my_sessions(payload: dict = Depends(get_current_staff_payload)):
    ensure_customer_service_tables()

    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    slp_id = payload["sub"]
    conn = get_db()
    cur = conn.cursor()

    # ✅ refresh priority by waiting minutes (only increase)
    cur.execute("""
        UPDATE service_sessions
        SET priority = MAX(
            priority,
            CASE
                WHEN CAST((julianday('now') - julianday(created_at)) * 24 * 60 AS INTEGER) <= 5 THEN 0
                WHEN CAST((julianday('now') - julianday(created_at)) * 24 * 60 AS INTEGER) <= 10 THEN 1
                ELSE 2
            END
        )
        WHERE status IN ('queued','assigned','open')
    """)
    conn.commit()

    cur.execute("""
        SELECT id, patient_id, topic, priority, status, created_at, last_message_at
        FROM service_sessions
        WHERE slp_id=? AND status IN ('assigned','open')  
        ORDER BY created_at ASC
    """, (slp_id,))
    rows = cur.fetchall()
    conn.close()

    return {
        "items": [
            {
                "id": r[0],
                "patient_id": r[1],
                "topic": r[2],
                "priority": r[3],
                "status": r[4],
                "created_at": r[5],
                "last_message_at": r[6],
            }
            for r in rows
        ]
    }


def _parse_ymd(s: str) -> str:
    # 回傳 YYYY-MM-DD（給 sqlite 用）
    try:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format, use YYYY-MM-DD")

@app.get("/api/slp/stats")
def api_slp_stats(
    date_from: str = Query(..., alias="from", description="YYYY-MM-DD"),
    date_to: str = Query(..., alias="to", description="YYYY-MM-DD"),
    payload: dict = Depends(get_current_staff_payload),
):
    ensure_customer_service_tables()

    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    slp_id = payload["sub"]
    d1 = _parse_ymd(date_from)
    d2 = _parse_ymd(date_to)

    conn = get_db()
    cur = conn.cursor()
    try:
        # -------------------------
        # 1) Sessions count (期間內 created_at)
        # -------------------------
        cur.execute("""
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN status='closed' THEN 1 ELSE 0 END) AS closed,
              SUM(CASE WHEN status IN ('assigned','open') THEN 1 ELSE 0 END) AS open_like
            FROM service_sessions
            WHERE slp_id=?
              AND date(created_at) BETWEEN date(?) AND date(?)
        """, (slp_id, d1, d2))
        r = cur.fetchone()
        sessions_total = int(r[0] or 0)
        sessions_closed = int(r[1] or 0)
        sessions_open_like = int(r[2] or 0)

        # -------------------------
        # 2) Avg close minutes (只算 closed 且 closed_at 有值)
        # -------------------------
        cur.execute("""
            SELECT
              AVG((julianday(closed_at) - julianday(created_at)) * 24 * 60)
            FROM service_sessions
            WHERE slp_id=?
              AND status='closed'
              AND closed_at IS NOT NULL
              AND date(closed_at) BETWEEN date(?) AND date(?)
        """, (slp_id, d1, d2))
        avg_close_minutes = cur.fetchone()[0]
        avg_close_minutes = float(avg_close_minutes) if avg_close_minutes is not None else None

        # -------------------------
        # 3) Messages count (期間內 message created_at)
        # -------------------------
        cur.execute("""
            SELECT
              SUM(CASE WHEN m.sender_type='slp' THEN 1 ELSE 0 END) AS slp_messages,
              SUM(CASE WHEN m.sender_type='patient' THEN 1 ELSE 0 END) AS patient_messages
            FROM service_messages m
            JOIN service_sessions s ON s.id = m.session_id
            WHERE s.slp_id=?
              AND date(m.created_at) BETWEEN date(?) AND date(?)
        """, (slp_id, d1, d2))
        r = cur.fetchone()
        slp_messages = int(r[0] or 0)
        patient_messages = int(r[1] or 0)

        # -------------------------
        # 4) Avg first response minutes
        # 定義：同一 session 中
        # - first_patient_at = MIN(patient msg time)
        # - first_slp_at = MIN(slp msg time)
        # 只計算 first_slp_at >= first_patient_at 且兩者皆存在
        # -------------------------
        cur.execute("""
            WITH firsts AS (
              SELECT
                s.id AS session_id,
                MIN(CASE WHEN m.sender_type='patient' THEN m.created_at END) AS first_patient_at,
                MIN(CASE WHEN m.sender_type='slp' THEN m.created_at END)     AS first_slp_at
              FROM service_sessions s
              JOIN service_messages m ON m.session_id = s.id
              WHERE s.slp_id=?
                AND date(s.created_at) BETWEEN date(?) AND date(?)
              GROUP BY s.id
            )
            SELECT AVG((julianday(first_slp_at) - julianday(first_patient_at)) * 24 * 60)
            FROM firsts
            WHERE first_patient_at IS NOT NULL
              AND first_slp_at IS NOT NULL
              AND julianday(first_slp_at) >= julianday(first_patient_at)
        """, (slp_id, d1, d2))
        avg_first_response_minutes = cur.fetchone()[0]
        avg_first_response_minutes = float(avg_first_response_minutes) if avg_first_response_minutes is not None else None

        return {
            "slp_id": slp_id,
            "range": {"from": d1, "to": d2},
            "sessions": {
                "total": sessions_total,
                "closed": sessions_closed,
                "open_like": sessions_open_like
            },
            "messages": {
                "slp": slp_messages,
                "patient": patient_messages
            },
            "timing": {
                "avg_first_response_minutes": avg_first_response_minutes,
                "avg_close_minutes": avg_close_minutes
            }
        }
    finally:
        conn.close()

@app.get("/api/slp/stats/daily")
def api_slp_stats_daily(
    date_from: str = Query(..., alias="from", description="YYYY-MM-DD"),
    date_to: str = Query(..., alias="to", description="YYYY-MM-DD"),
    payload: dict = Depends(get_current_staff_payload),
):
    ensure_customer_service_tables()

    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    slp_id = payload["sub"]
    d1 = _parse_ymd(date_from)
    d2 = _parse_ymd(date_to)

    conn = get_db()
    cur = conn.cursor()
    try:
        # 以 session.created_at 當作「該單屬於哪天」的統計基準
        # closed_count：用 closed_at 落在該日的數量（較符合「哪天結案」）
        # messages：用 message.created_at 落在該日的數量（訊息發生在哪天）
        cur.execute("""
            WITH RECURSIVE dates(d) AS (
              SELECT date(?)
              UNION ALL
              SELECT date(d, '+1 day') FROM dates WHERE d < date(?)
            ),
            closed AS (
              SELECT date(closed_at) AS d, COUNT(*) AS closed_count
              FROM service_sessions
              WHERE slp_id=?
                AND status='closed'
                AND closed_at IS NOT NULL
                AND date(closed_at) BETWEEN date(?) AND date(?)
              GROUP BY date(closed_at)
            ),
            msgs AS (
              SELECT date(m.created_at) AS d,
                     SUM(CASE WHEN m.sender_type='slp' THEN 1 ELSE 0 END) AS slp_messages,
                     SUM(CASE WHEN m.sender_type='patient' THEN 1 ELSE 0 END) AS patient_messages
              FROM service_messages m
              JOIN service_sessions s ON s.id=m.session_id
              WHERE s.slp_id=?
                AND date(m.created_at) BETWEEN date(?) AND date(?)
              GROUP BY date(m.created_at)
            )
            SELECT
              dates.d,
              COALESCE(closed.closed_count, 0) AS closed_count,
              COALESCE(msgs.slp_messages, 0) AS slp_messages,
              COALESCE(msgs.patient_messages, 0) AS patient_messages
            FROM dates
            LEFT JOIN closed ON closed.d = dates.d
            LEFT JOIN msgs   ON msgs.d   = dates.d
            ORDER BY dates.d ASC
        """, (d1, d2, slp_id, d1, d2, slp_id, d1, d2))

        items = [
            {
                "date": r[0],
                "closed_count": int(r[1] or 0),
                "slp_messages": int(r[2] or 0),
                "patient_messages": int(r[3] or 0),
            }
            for r in cur.fetchall()
        ]

        return {"slp_id": slp_id, "range": {"from": d1, "to": d2}, "items": items}
    finally:
        conn.close()

@app.post("/api/system/allocate")
def api_system_allocate():
    ensure_customer_service_tables()
    out = system_auto_allocate_one_db()
    return {"result": out}


# ==============================
# ✅ Staff Update Profile API
# ==============================

class StaffUpdateProfileRequest(BaseModel):
    username: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    license_number: Optional[str] = None
    organization: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None

# ==============================
# ✅ Staff Update Profile API
# ==============================

class StaffUpdateProfileRequest(BaseModel):
    username: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    license_number: Optional[str] = None
    organization: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    password: Optional[str] = None          # ✅ allow password update (>=6) without old password? (we will NOT use here)
    old_password: Optional[str] = None      # ✅ optional (not used here)
    new_password: Optional[str] = None      # ✅ optional (not used here)

@app.post("/staff/update-profile")
def staff_update_profile(
    data: StaffUpdateProfileRequest,
    payload: dict = Depends(get_current_staff_payload),
):
    """
    Staff 自己更新個人資料（給 SLP Settings 用）
    - 用 staffToken 驗證（payload.staff==1）
    - email 若變更：強制 verify=0 並寄出新的驗證碼
    - 密碼更新：請走 /staff/change-password（避免混在同一支）
    """
    staff_id = payload.get("sub")
    role_from_token = payload.get("role")

    if not staff_id:
        raise HTTPException(status_code=401, detail="Invalid staff token")

    # 不允許用 update-profile 改密碼（避免安全問題）
    if data.password is not None or data.old_password is not None or data.new_password is not None:
        raise HTTPException(status_code=400, detail="Use /staff/change-password to update password")

    allowed_fields = {
        "username": data.username,
        "name": data.name,
        "email": data.email,
        "license_number": data.license_number,
        "organization": data.organization,
        "description": data.description,
        "language": data.language,
    }
    patch = {k: v for k, v in allowed_fields.items() if v is not None}

    if not patch:
        return {"msg": "No fields updated", "staff": None}

    # 基本格式檢查
    if "email" in patch:
        em = str(patch["email"]).strip()
        # ✅ enforce gmail only
        if not re.match(r"^[A-Za-z0-9._%+-]+@gmail\.com$", em, re.IGNORECASE):
            raise HTTPException(status_code=400, detail="Email must be ...@gmail.com")
        patch["email"] = em

    if "username" in patch:
        patch["username"] = str(patch["username"]).strip()
        if not patch["username"]:
            raise HTTPException(status_code=400, detail="Username cannot be empty")

    conn = get_db()
    cur = conn.cursor()
    try:
        # 先查目前 staff 資料
        cur.execute("SELECT id, username, email, role, verify FROM staff_users WHERE id=?", (staff_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Staff not found")

        current_email = row[2] or ""
        current_role = row[3]
        current_verify = int(row[4] or 0)

        # username unique
        if "username" in patch:
            cur.execute("SELECT 1 FROM staff_users WHERE username=? AND id<>?", (patch["username"], staff_id))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="Username already exists")

        # email+role unique
        if "email" in patch:
            cur.execute(
                "SELECT 1 FROM staff_users WHERE email=? AND role=? AND id<>?",
                (patch["email"], current_role, staff_id)
            )
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="Email already exists for this role")

        # ✅ if email changes => force verify=0 and send verify code
        email_changed = ("email" in patch) and (patch["email"].lower() != str(current_email).lower())

        # 組 UPDATE
        updates = []
        params = []
        for k, v in patch.items():
            updates.append(f"{k}=?")
            params.append(v)

        if email_changed:
            updates.append("verify=?")
            params.append(0)

        sql = f"UPDATE staff_users SET {', '.join(updates)} WHERE id=?"
        params.append(staff_id)
        cur.execute(sql, tuple(params))
        conn.commit()

        # 若 email 變更 => 建立/更新驗證碼並寄信
        if email_changed:
            ensure_staff_verify_table()
            verify_code = str(random.randint(100000, 999999))

            # store code by (email, role)
            cur.execute(
                "INSERT OR REPLACE INTO staff_email_verifies (email, role, code, created_at) VALUES (?, ?, ?, ?)",
                (patch["email"], current_role, verify_code, datetime.now())
            )
            conn.commit()

            # send email (reuse your Gmail settings)
            try:
                sender = "aaa2025819@gmail.com"
                smtp_host = "smtp.gmail.com"
                smtp_port = 587
                smtp_user = "aaa2025819@gmail.com"
                smtp_pass = "ssjuayuhpfugwltp"

                subject = "Staff Email Verification Code"
                body = f"""
Hello {row[1]},

Your staff verification code is: {verify_code}

Role: {current_role}

You changed your email, so verification is required again.

Thank you,
ASR+LLM System
"""
                msg = MIMEText(body, "plain", "utf-8")
                msg["Subject"] = Header(subject, "utf-8")
                msg["From"] = sender
                msg["To"] = patch["email"]

                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.sendmail(sender, [patch["email"]], msg.as_string())

            except Exception as e:
                # email sending failure should be visible
                raise HTTPException(status_code=500, detail=f"Email send failed: {e}")

        # audit log
        try:
            audit_log(
                actor_type=role_from_token if role_from_token in ("slp", "admin") else "slp",
                actor_id=staff_id,
                action="STAFF_UPDATE_PROFILE",
                resource_type="staff_users",
                resource_id=staff_id,
                metadata=json.dumps({"patch": patch, "email_changed": email_changed}, ensure_ascii=False),
            )
        except Exception:
            pass

        # 回傳更新後資料
        cur.execute("""
            SELECT id, username, name, email, role, license_number, organization,
                   description, language, verify, is_approved, created_at, staff_picture
            FROM staff_users WHERE id=?
        """, (staff_id,))
        s = cur.fetchone()

        return {
            "msg": "Profile updated successfully",
            "staff": {
                "id": s[0],
                "username": s[1],
                "name": s[2],
                "email": s[3],
                "role": s[4],
                "license_number": s[5],
                "organization": s[6],
                "description": s[7],
                "language": s[8],
                "verify": s[9],
                "is_approved": s[10],
                "created_at": s[11],
                "staff_picture": s[12],
            }
        }

    finally:
        conn.close()


class StaffChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

@app.post("/staff/change-password")
def staff_change_password(
    data: StaffChangePasswordRequest,
    payload: dict = Depends(get_current_staff_payload)
):
    """
    Staff 已登入改密碼
    - 驗 old_password
    - new_password 長度 >= 6
    """
    staff_id = payload.get("sub")
    if not staff_id:
        raise HTTPException(status_code=401, detail="Invalid staff token")

    if not data.new_password or len(data.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT password_hash FROM staff_users WHERE id=?", (staff_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Staff not found")

        if not verify_password(data.old_password, row[0]):
            raise HTTPException(status_code=400, detail="Old password is incorrect")

        cur.execute(
            "UPDATE staff_users SET password_hash=? WHERE id=?",
            (hash_password(data.new_password), staff_id)
        )
        conn.commit()

        return {"msg": "✅ Password updated successfully"}
    finally:
        conn.close()

@app.delete("/staff/delete-picture")
def delete_staff_picture(
    payload: dict = Depends(get_current_staff_payload)
):
    """
    刪除 staff 頭像：
    - 從 User/staff/<staff_id>/picture 刪除檔案
    - DB staff_users.staff_picture 設為 NULL
    """
    staff_id = payload.get("sub")
    if not staff_id:
        raise HTTPException(status_code=401, detail="Invalid staff token")

    staff_dir = os.path.join(USER_ROOT_DIR, "staff", str(staff_id), "picture")

    # 1) delete files on disk
    if os.path.exists(staff_dir):
        for fn in os.listdir(staff_dir):
            try:
                os.remove(os.path.join(staff_dir, fn))
            except Exception:
                pass

    # 2) update DB
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE staff_users SET staff_picture=NULL WHERE id=?", (staff_id,))
        conn.commit()
    finally:
        conn.close()

    return {"msg": "✅ Staff avatar deleted"}
        
@app.get("/api/slp/session/{session_id}")
def api_slp_get_one_session(
    session_id: str,
    payload: dict = Depends(get_current_staff_payload)
):
    ensure_customer_service_tables()

    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    slp_id = payload["sub"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, patient_id, slp_id, topic, priority, status, created_at, last_message_at, closed_at, closed_by
        FROM service_sessions
        WHERE id=?
    """, (session_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="session not found")

    # ✅ 只能看分配給自己的單
    if row[2] != slp_id:
        raise HTTPException(status_code=403, detail="not allowed")

    return {
        "id": row[0],
        "patient_id": row[1],
        "slp_id": row[2],
        "topic": row[3],
        "priority": row[4],
        "status": row[5],
        "created_at": row[6],
        "last_message_at": row[7],
        "closed_at": row[8],
        "closed_by": row[9],
    }

@app.get("/api/slp/session/{session_id}/messages")
def api_slp_get_session_messages(
    session_id: str,
    payload: dict = Depends(get_current_staff_payload)
):
    ensure_customer_service_tables()

    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    slp_id = payload["sub"]
    conn = get_db()
    cur = conn.cursor()

    # ✅ 確認此 session 屬於該 SLP
    cur.execute("SELECT slp_id FROM service_sessions WHERE id=?", (session_id,))
    srow = cur.fetchone()
    if not srow:
        conn.close()
        raise HTTPException(status_code=404, detail="session not found")
    if srow[0] != slp_id:
        conn.close()
        raise HTTPException(status_code=403, detail="not allowed")

    cur.execute("""
        SELECT id, sender_type, sender_id, message, message_type, created_at
        FROM service_messages
        WHERE session_id=?
        ORDER BY created_at ASC
    """, (session_id,))
    items = [
        {
            "id": r[0],
            "sender_type": r[1],
            "sender_id": r[2],
            "message": r[3],
            "message_type": r[4],
            "created_at": r[5],
        }
        for r in cur.fetchall()
    ]
    conn.close()
    return {"items": items}

class PostSlpServiceMessageReq(BaseModel):
    message: str

@app.post("/api/slp/session/{session_id}/messages")
def api_slp_post_session_message(
    session_id: str,
    data: PostSlpServiceMessageReq,
    payload: dict = Depends(get_current_staff_payload)
):
    ensure_customer_service_tables()

    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    slp_id = payload["sub"]
    msg = (data.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is empty")

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT slp_id, status FROM service_sessions WHERE id=?", (session_id,))
        srow = cur.fetchone()
        if not srow:
            raise HTTPException(status_code=404, detail="session not found")
        if srow[0] != slp_id:
            raise HTTPException(status_code=403, detail="not allowed")
        if srow[1] in ("closed", "cancelled"):
            raise HTTPException(status_code=400, detail="session already closed")

        msg_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO service_messages (id, session_id, sender_type, sender_id, message, message_type)
            VALUES (?, ?, 'slp', ?, ?, 'text')
        """, (msg_id, session_id, slp_id, msg))

        # ✅ update last message + last slp activity + open
        cur.execute("""
            UPDATE service_sessions
            SET last_message_at=CURRENT_TIMESTAMP,
                last_slp_activity_at=CURRENT_TIMESTAMP,
                status=CASE WHEN status='assigned' THEN 'open' ELSE status END
            WHERE id=?
        """, (session_id,))

        conn.commit()

        audit_log("slp", slp_id, "POST_SERVICE_MESSAGE", "service_message", msg_id,
                  metadata=json.dumps({"session_id": session_id}, ensure_ascii=False))

        return {"message_id": msg_id}
    finally:
        conn.close()

@app.post("/api/slp/session/{session_id}/close")
def api_slp_close_session(
    session_id: str,
    payload: dict = Depends(get_current_staff_payload)
):
    ensure_customer_service_tables()

    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    slp_id = payload["sub"]
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT slp_id, status FROM service_sessions WHERE id=?", (session_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="session not found")
        if row[0] != slp_id:
            raise HTTPException(status_code=403, detail="not allowed")
        if row[1] in ("closed", "cancelled"):
            return {"msg": "already closed"}

        # ✅ 強制規則：病人 5 分鐘未回覆 SLP 才能關
        if not patient_inactive_after_slp_reply(cur, session_id, SLP_INACTIVE_CLOSE_MINUTES):
            raise HTTPException(
                status_code=400,
                detail=f"病人尚未達到 {SLP_INACTIVE_CLOSE_MINUTES} 分鐘未回覆，不能關閉。"
            )

        cur.execute("""
            UPDATE service_sessions
            SET status='closed', closed_at=CURRENT_TIMESTAMP, closed_by='slp'
            WHERE id=?
        """, (session_id,))

        # ✅ 寫入 system_event（讓聊天紀錄可追溯）
        sys_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO service_messages (id, session_id, sender_type, sender_id, message, message_type)
            VALUES (?, ?, 'system', NULL, ?, 'system_event')
        """, (sys_id, session_id, f"SLP 因病人逾時未回覆（{SLP_INACTIVE_CLOSE_MINUTES} 分鐘）而關閉此服務。"))

        conn.commit()

        audit_log("slp", slp_id, "CLOSE_SERVICE_SESSION", "service_session", session_id,
                  metadata=json.dumps({"rule": "patient_inactive", "minutes": SLP_INACTIVE_CLOSE_MINUTES}, ensure_ascii=False))

        return {"msg": "closed_due_to_inactivity"}
    finally:
        conn.close()

@app.post("/api/slp/session/{session_id}/close-inactive")
def api_slp_close_session_inactive(
    session_id: str,
    payload: dict = Depends(get_current_staff_payload)
):
    ensure_customer_service_tables()

    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    slp_id = payload["sub"]
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT slp_id, status FROM service_sessions WHERE id=?", (session_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="session not found")
        if row[0] != slp_id:
            raise HTTPException(status_code=403, detail="not allowed")
        if row[1] in ("closed", "cancelled"):
            return {"msg": "already closed"}

        # ✅ 核心規則：病人 5 分鐘未回覆 SLP 才能關
        if not patient_inactive_after_slp_reply(cur, session_id, SLP_INACTIVE_CLOSE_MINUTES):
            raise HTTPException(
                status_code=400,
                detail=f"病人尚未達到 {SLP_INACTIVE_CLOSE_MINUTES} 分鐘未回覆，不能用逾時關單。"
            )

        cur.execute("""
            UPDATE service_sessions
            SET status='closed', closed_at=CURRENT_TIMESTAMP, closed_by='slp'
            WHERE id=?
        """, (session_id,))

        # （可選）插入 system_event 訊息，讓聊天紀錄看到原因
        sys_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO service_messages (id, session_id, sender_type, sender_id, message, message_type)
            VALUES (?, ?, 'system', NULL, ?, 'system_event')
        """, (sys_id, session_id, f"SLP 因病人逾時未回覆（{SLP_INACTIVE_CLOSE_MINUTES} 分鐘）而關閉此服務。"))

        conn.commit()

        audit_log("slp", slp_id, "CLOSE_SERVICE_SESSION_INACTIVE", "service_session", session_id,
                  metadata=json.dumps({"minutes": SLP_INACTIVE_CLOSE_MINUTES}, ensure_ascii=False))

        return {"msg": "closed_due_to_inactivity"}
    finally:
        conn.close()

@app.get("/api/slp/patient/{patient_id}/tests/stats")
def api_slp_patient_tests_stats(
    patient_id: str,
    days: int = Query(365, ge=1, le=3650),
    date_from: Optional[str] = Query(None, alias="from"),
    date_to: Optional[str] = Query(None, alias="to"),
    language: Optional[str] = Query(None, description="cn|en (optional)"),
    a: Optional[str] = Query(None),
    b: Optional[str] = Query(None),
    payload: dict = Depends(get_current_staff_payload),
):
    # 1) auth: only slp
    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    slp_id = payload["sub"]

    # 2) permission: slp must have sessions with this patient
    ensure_customer_service_tables()
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 1
            FROM service_sessions
            WHERE patient_id=? AND slp_id=? AND status IN ('assigned','open','closed')
            LIMIT 1
        """, (patient_id, slp_id))
        if not cur.fetchone():
            raise HTTPException(status_code=403, detail="not allowed")

        # 3) Now run the SAME logic as /api/patient/tests/stats,
        #    but using patient_id as the target user.
        # ---- copy of api_patient_tests_stats body starts ----
        d1_def, d2_def = _ymd_default_range(days)
        d1 = _parse_ymd_or_default(date_from, d1_def)
        d2 = _parse_ymd_or_default(date_to, d2_def)

        where_lang = ""
        params: list[Any] = [patient_id, d1, d2]
        if language is not None and str(language).strip() != "":
            lang = normalize_practice_language(language)  # cn|en
            where_lang = " AND language=?"
            params.append(lang)

        cur.execute(f"""
            SELECT
              id, test_name, status, language, threshold, words_per_subtype,
              progress_cursor, total_questions, current_question_index,
              has_slp, date, updated_at, finished_at, result
            FROM tests
            WHERE patient_id=?
              AND status='finished'
              AND date(COALESCE(finished_at, updated_at, date)) BETWEEN date(?) AND date(?)
              {where_lang}
            ORDER BY datetime(COALESCE(finished_at, updated_at, date)) DESC
        """, tuple(params))
        tests = cur.fetchall()

        items: List[dict] = []
        for t in tests:
            test_id = t["id"]

            cur.execute("""
                SELECT score, correct_rate, severity, check_results
                FROM test_records
                WHERE test_id=?
            """, (test_id,))
            rec_rows = cur.fetchall()
            recordsAgg = _test_agg_from_rows(rec_rows)

            cur.execute("""
                SELECT super_type, sub_type
                FROM slp_patient
                WHERE test_id=?
            """, (test_id,))
            slp_rows = cur.fetchall()
            slpAgg = _slp_agg_from_rows(slp_rows)

            items.append({
                "test": {
                    "id": test_id,
                    "test_name": t["test_name"],
                    "status": t["status"],
                    "language": t["language"],
                    "threshold": int(t["threshold"] or 0),
                    "words_per_subtype": int(t["words_per_subtype"] or 0),
                    "progress_cursor": int(t["progress_cursor"] or 0),
                    "total_questions": int(t["total_questions"] or 0),
                    "current_question_index": int(t["current_question_index"] or 0),
                    "has_slp": int(t["has_slp"] or 0),
                    "created_at": t["date"],
                    "updated_at": t["updated_at"],
                    "finished_at": t["finished_at"],
                    "result": _safe_json_loads(t["result"]),
                },
                "recordsAgg": recordsAgg,
                "slpAgg": slpAgg,
            })

        latest = items[0] if len(items) >= 1 else None
        prev = items[1] if len(items) >= 2 else None

        default_compare = None
        if latest and prev:
            default_compare = {
                "a": prev["test"]["id"],
                "b": latest["test"]["id"],
                "delta": _compare_metrics(prev, latest),
            }

        custom_compare = None
        if a and b:
            map_by_id = {it["test"]["id"]: it for it in items}
            if a in map_by_id and b in map_by_id:
                A = map_by_id[a]
                B = map_by_id[b]
                custom_compare = {"a": a, "b": b, "delta": _compare_metrics(A, B)}

        return {
            "range": {"from": d1, "to": d2},
            "items": items,
            "latest_finished": latest,
            "prev_finished": prev,
            "default_compare": default_compare,
            "custom_compare": custom_compare,
        }
        # ---- copy ends ----
    finally:
        conn.close()
        
@app.get("/api/slp/patient/{patient_id}/summary")
def api_slp_patient_summary(
    patient_id: str,
    limit: int = Query(50),
    payload: dict = Depends(get_current_staff_payload)
):
    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    ensure_customer_service_tables()

    slp_id = payload["sub"]
    conn = get_db()
    cur = conn.cursor()
    try:
        # ✅ 權限：只能查「分配給自己」的 session 中出現的病人
        cur.execute("""
            SELECT 1
            FROM service_sessions
            WHERE patient_id=? AND slp_id=? AND status IN ('assigned','open','closed')
            LIMIT 1
        """, (patient_id, slp_id))
        if not cur.fetchone():
            raise HTTPException(status_code=403, detail="not allowed")

        # 1) patient 基本資料 + task_id
        cur.execute("""
            SELECT id, username, name, age, diagnosis, language, email, task_id
            FROM patients
            WHERE id=?
        """, (patient_id,))
        p = cur.fetchone()
        if not p:
            raise HTTPException(status_code=404, detail="patient not found")

        patient = {
            "id": p[0],
            "username": p[1],
            "name": p[2],
            "age": p[3],
            "diagnosis": p[4],
            "language": p[5],
            "email": p[6],
            "task_id": p[7],
        }

        # 2) task 詳細（若有）
        task = None
        if patient["task_id"]:
            cur.execute("""
                SELECT id, title, description, category, due_date, total_times, progress_times, integral, status, created_at
                FROM tasks
                WHERE id=?
            """, (patient["task_id"],))
            t = cur.fetchone()
            if t:
                task = {
                    "id": t[0],
                    "title": t[1],
                    "description": t[2],
                    "category": t[3],
                    "due_date": t[4],
                    "total_times": t[5],
                    "progress_times": t[6],
                    "integral": t[7],
                    "status": t[8],
                    "created_at": t[9],
                }

        # 3) 訓練紀錄（CN/EN 都列出，簡單 list）
        cur.execute("""
            SELECT date, language, test_type, category, target_word, asr_word, correct_rate, difficulty
            FROM records_analysis
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT ?
        """, (patient_id, int(limit)))
        records = [
            {
                "date": r[0],
                "language": r[1],
                "test_type": r[2],
                "category": r[3],
                "target_word": r[4],
                "asr_word": r[5],
                "correct_rate": r[6],
                "difficulty": r[7],
            }
            for r in cur.fetchall()
        ]

        return {"patient": patient, "task": task, "records": records}

    finally:
        conn.close()

@app.get("/api/slp/sessions/history")
def api_slp_sessions_history(
    status: str = Query("all", description="all|assigned|open|closed|cancelled"),
    limit: int = Query(80),
    payload: dict = Depends(get_current_staff_payload)
):
    ensure_customer_service_tables()

    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    slp_id = payload["sub"]

    status = (status or "all").strip().lower()
    allowed = {"all", "assigned", "open", "closed", "cancelled"}
    if status not in allowed:
        raise HTTPException(status_code=400, detail="Invalid status")

    conn = get_db()
    cur = conn.cursor()
    try:
        where_status = ""
        params = [slp_id]

        if status != "all":
            where_status = " AND status=?"
            params.append(status)

        # 取 session + 病人 id + topic + priority + 狀態 + 建立/最後訊息/關閉時間
        cur.execute(f"""
            SELECT id, patient_id, topic, priority, status, created_at, last_message_at, closed_at, closed_by
            FROM service_sessions
            WHERE slp_id=?
            {where_status}
            ORDER BY COALESCE(last_message_at, created_at) DESC
            LIMIT ?
        """, (*params, int(limit)))

        rows = cur.fetchall()

        return {
            "items": [
                {
                    "id": r[0],
                    "patient_id": r[1],
                    "topic": r[2],
                    "priority": r[3],
                    "status": r[4],
                    "created_at": r[5],
                    "last_message_at": r[6],
                    "closed_at": r[7],
                    "closed_by": r[8],
                }
                for r in rows
            ]
        }
    finally:
        conn.close()

class UpdateUsernameRequest(BaseModel):
    username: str

@app.patch("/api/patient/update-username")
def update_username(
    data: UpdateUsernameRequest,
    current_user_id: str = Depends(get_current_user_id),
):
    new_username = (data.username or "").strip()

    if not new_username:
        raise HTTPException(status_code=400, detail="Username cannot be empty")

    # 你也可以加更嚴格規則（只允許英文數字底線）
    if len(new_username) < 3 or len(new_username) > 24:
        raise HTTPException(status_code=400, detail="Username must be 3-24 characters")

    conn = get_db()
    cur = conn.cursor()

    # 檢查是否被其他人使用
    cur.execute("SELECT 1 FROM patients WHERE username=? AND id<>?", (new_username, current_user_id))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")

    # 更新自己的 username
    cur.execute("UPDATE patients SET username=? WHERE id=?", (new_username, current_user_id))
    conn.commit()

    # 回傳最新資料
    cur.execute("""
        SELECT id, username, name, age, diagnosis, email, language, integral, verify, user_picture
        FROM patients WHERE id=?
    """, (current_user_id,))
    row = cur.fetchone()
    conn.close()

    return {
        "msg": "Username updated successfully",
        "user": {
            "id": row[0],
            "username": row[1],
            "name": row[2],
            "age": row[3],
            "diagnosis": row[4],
            "email": row[5],
            "language": row[6],
            "integral": row[7],
            "verify": row[8],
            "user_picture": row[9],
        }
    }    

from pydantic import BaseModel
from fastapi import Query

class PracticeProgressResp(BaseModel):
    super_type: str
    sub_type: str
    language: str
    chunk_size: int
    next_start_index: int
    next_range_start: int   # 1-based for UI
    next_range_end: int     # 1-based for UI
    last_range_end: int

@app.get("/api/practice/progress", response_model=PracticeProgressResp)
def get_practice_progress(
    super_type: str = Query(...),
    sub_type: str = Query(...),
    language: str = Query("en"),
    chunk_size: int = Query(50),
    current_user_id: str = Depends(get_current_user_id),
):
    ensure_practice_progress_table()

    lang = normalize_practice_language(language)

    chunk_size = int(chunk_size)
    if chunk_size <= 0 or chunk_size > 200:
        raise HTTPException(status_code=400, detail="Invalid chunk_size")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT chunk_size, next_start_index, last_range_end
        FROM practice_progress
        WHERE user_id=? AND super_type=? AND sub_type=? AND language=?
    """, (current_user_id, super_type, sub_type, lang))
    row = cur.fetchone()
    conn.close()

    if row:
        cs, next_start_index, last_range_end = int(row[0]), int(row[1]), int(row[2] or 0)
    else:
        cs, next_start_index, last_range_end = chunk_size, 0, 0

    next_range_start = next_start_index + 1
    next_range_end = next_start_index + cs

    return PracticeProgressResp(
        super_type=super_type,
        sub_type=sub_type,
        language=lang,
        chunk_size=cs,
        next_start_index=next_start_index,
        next_range_start=next_range_start,
        next_range_end=next_range_end,
        last_range_end=last_range_end,
    )

class CommitRangeReq(BaseModel):
    super_type: str
    sub_type: str
    language: str = "en"
    chunk_size: int = 50
    total_items: int

class CommitRangeResp(BaseModel):
    committed_range_start: int  # 1-based
    committed_range_end: int    # 1-based
    next_start_index: int

@app.post("/api/practice/commit-range", response_model=CommitRangeResp)
def commit_practice_range(
    data: CommitRangeReq,
    current_user_id: str = Depends(get_current_user_id),
):
    ensure_practice_progress_table()

    super_type = (data.super_type or "").strip()
    sub_type = (data.sub_type or "").strip()
    lang = normalize_practice_language(data.language)

    chunk_size = int(data.chunk_size or 50)
    total_items = int(data.total_items or 0)

    if not super_type or not sub_type:
        raise HTTPException(status_code=400, detail="Missing super_type/sub_type")
    if chunk_size <= 0 or chunk_size > 200:
        raise HTTPException(status_code=400, detail="Invalid chunk_size")
    if total_items <= 0:
        raise HTTPException(status_code=400, detail="Invalid total_items")

    conn = get_db()
    cur = conn.cursor()
    try:
        # ✅ 用 language 定位
        cur.execute("""
            SELECT id, next_start_index
            FROM practice_progress
            WHERE user_id=? AND super_type=? AND sub_type=? AND language=?
        """, (current_user_id, super_type, sub_type, lang))
        row = cur.fetchone()

        if row:
            prog_id, next_start_index = row[0], int(row[1] or 0)
        else:
            prog_id, next_start_index = str(uuid.uuid4()), 0
            cur.execute("""
                INSERT INTO practice_progress
                (id, user_id, super_type, sub_type, language, chunk_size, next_start_index, last_range_end)
                VALUES (?, ?, ?, ?, ?, ?, 0, 0)
            """, (prog_id, current_user_id, super_type, sub_type, lang, chunk_size))

        committed_start_idx = next_start_index
        committed_end_idx_excl = min(next_start_index + chunk_size, total_items)  # exclusive

        if committed_start_idx >= total_items:
            raise HTTPException(status_code=400, detail="No more items to commit")

        committed_range_start = committed_start_idx + 1
        committed_range_end = committed_end_idx_excl  # 1-based end
        new_next_start_index = committed_end_idx_excl

        cur.execute("""
            UPDATE practice_progress
            SET chunk_size=?,
                next_start_index=?,
                last_range_end=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (chunk_size, new_next_start_index, committed_range_end, prog_id))

        conn.commit()

        return CommitRangeResp(
            committed_range_start=committed_range_start,
            committed_range_end=committed_range_end,
            next_start_index=new_next_start_index,
        )

    finally:
        conn.close()

class NextWordsResp(BaseModel):
    super_type: str
    sub_type: str
    language: str
    range_start: int   # 1-based
    range_end: int     # 1-based
    total_items: int
    items: list[str]

@app.get("/api/practice/instruction")
def practice_instruction(
    super_type: str = Query(...),
    sub_type: str = Query(...),
):
    # 先給你最簡單版本：固定模板
    # 之後你也可以針對不同 sub_type 寫更細規則
    return {
        "super_type": super_type,
        "sub_type": sub_type,
        "how_to_read": [
            "1) 先看目標單字（target word）。",
            "2) 按播放或看提示，跟著唸一次。",
            "3) 按錄音開始，清楚唸出單字。",
            "4) 停止錄音並送出分析。",
        ],
        "show_answer_tip": "按「顯示答案」可以看到標準發音/提示（若有）。"
    }

class PracticeSubmitReq(BaseModel):
    super_type: str
    sub_type: str
    range_start: int  # 1-based
    range_end: int    # 1-based
    items: list[dict] # e.g. [{w:"rescued", ipa:"/ˈɹɛskjud/", correct:true, ...}, ...]

@app.post("/api/practice/submit")
def practice_submit(
    data: PracticeSubmitReq,
    current_user_id: str = Depends(get_current_user_id),
):
    # 這裡先只存檔，不做 ASR 分析（因為你現有分析是走 /analyze_*_words）
    payload = {
        "super_type": data.super_type,
        "sub_type": data.sub_type,
        "range": {"start": data.range_start, "end": data.range_end},
        "items": data.items,
    }

    # 你已有 save_analysis_record，但參數不完全符合英文/這種練習
    # 這裡簡化：直接寫 records_analysis（用 SQL）
    conn = get_db()
    cur = conn.cursor()
    rid = str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        INSERT INTO records_analysis
        (id, user_id, date, target_word, target_ipa, asr_ipa, asr_word, score, correct_rate,
         difficulty, check_results, language, test_type, category, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        rid, current_user_id, now,
        None, None, None, None,
        None, None,
        "easy",  # 你 schema 限制 easy/medium/hard；這裡先給 easy
        json.dumps(payload, ensure_ascii=False),
        "English",
        "PracticeBank",
        data.super_type,
        now
    ))
    conn.commit()
    conn.close()

    return {"msg": "saved", "record_id": rid}

def get_current_user_level(current_user_id: str = Depends(get_current_user_id)) -> dict:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT age FROM patients WHERE id=?", (current_user_id,))
    row = cur.fetchone()
    conn.close()

    age = row[0] if row else None
    try:
        age_int = int(age) if age is not None else None
    except Exception:
        age_int = None

    # default if age is missing
    if age_int is None:
        level = "primary"
    elif age_int < 12:
        level = "primary"
    elif age_int < 18:
        level = "secondary"
    else:
        level = "advanced"

    return {"age": age_int, "level": level}

@app.get("/api/patient/level")
def api_patient_level(payload: dict = Depends(get_current_user_level)):
    return payload

@app.get("/admin/staff-applications")
def admin_list_staff_applications(payload: dict = Depends(get_current_staff_payload)):
    require_admin(payload)

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, username, name, email, role, license_number, organization, description, language,
                   verify, is_approved, created_at
            FROM staff_users
            WHERE is_approved=0
            ORDER BY created_at ASC
        """)
        rows = cur.fetchall()

        items = []
        for r in rows:
            items.append({
                "id": r[0],
                "username": r[1],
                "name": r[2],
                "email": r[3],
                "role": r[4],
                "license_number": r[5],
                "organization": r[6],
                "description": r[7],
                "language": r[8],
                "verify": r[9],
                "is_approved": r[10],
                "created_at": r[11],
            })

        return {"items": items}
    finally:
        conn.close()

class AdminDecideStaffApplicationReq(BaseModel):
    staff_id: str
    decision: str  # "approve" | "reject"
    reason: Optional[str] = None

@app.post("/admin/staff-applications/decide")
def admin_decide_staff_application(
    data: AdminDecideStaffApplicationReq,
    payload: dict = Depends(get_current_staff_payload)
):
    require_admin(payload)

    staff_id = (data.staff_id or "").strip()
    decision = (data.decision or "").strip().lower()
    reason = (data.reason or "").strip()

    if not staff_id:
        raise HTTPException(status_code=400, detail="Missing staff_id")
    if decision not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="Invalid decision")
    if decision == "reject" and not reason:
        raise HTTPException(status_code=400, detail="Reject reason is required")

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, username, email, role, is_approved
            FROM staff_users
            WHERE id=?
        """, (staff_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Staff not found")

        _id, username, email, role, is_approved = row
        if int(is_approved) != 0:
            raise HTTPException(status_code=400, detail="This application is not pending")

        if not email:
            raise HTTPException(status_code=400, detail="Staff email is missing")

        new_status = 1 if decision == "approve" else 2
        cur.execute("UPDATE staff_users SET is_approved=? WHERE id=?", (new_status, staff_id))
        conn.commit()

        # --- send email ---
        sender = "aaa2025819@gmail.com"
        smtp_host = "smtp.gmail.com"
        smtp_port = 587
        smtp_user = "aaa2025819@gmail.com"
        smtp_pass = "ssjuayuhpfugwltp"

        if decision == "approve":
            subject = "Your staff application has been approved"
            body = f"""
Hello {username},

Your {role.upper()} staff application has been approved.

You can now login with your email and password.

Thank you,
ASR+LLM System
"""
        else:
            subject = "Your staff application has been rejected"
            body = f"""
Hello {username},

Your {role.upper()} staff application has been rejected.

Reason:
{reason}

You may update your information and apply again if needed.

Thank you,
ASR+LLM System
"""

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = sender
        msg["To"] = email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(sender, [email], msg.as_string())

        # optional audit
        try:
            audit_log(
                actor_type="admin",
                actor_id=payload.get("sub"),
                action="ADMIN_DECIDE_STAFF_APPLICATION",
                resource_type="staff_users",
                resource_id=staff_id,
                metadata=json.dumps({"decision": decision, "reason": reason}, ensure_ascii=False),
            )
        except Exception:
            pass

        return {"msg": "ok", "staff_id": staff_id, "is_approved": new_status}

    finally:
        conn.close()


from typing import List, Dict, Any

def _date_series(d1: str, d2: str) -> List[str]:
    """
    Return list of YYYY-MM-DD between [d1, d2]
    """
    start = datetime.strptime(d1, "%Y-%m-%d").date()
    end = datetime.strptime(d2, "%Y-%m-%d").date()
    out = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d"))
        cur = cur + timedelta(days=1)
    return out

@app.get("/admin/stats/overview")
def admin_stats_overview(
    date_from: str = Query(..., alias="from", description="YYYY-MM-DD"),
    date_to: str = Query(..., alias="to", description="YYYY-MM-DD"),
    payload: dict = Depends(get_current_staff_payload),
):
    require_admin(payload)

    d1 = _parse_ymd(date_from)
    d2 = _parse_ymd(date_to)

    conn = get_db()
    cur = conn.cursor()
    try:
        # -------------------------
        # Totals (ALL TIME)
        # -------------------------
        cur.execute("SELECT COUNT(*) FROM patients")
        total_patients = int(cur.fetchone()[0] or 0)

        # only approved staff
        cur.execute("SELECT COUNT(*) FROM staff_users WHERE role='slp' AND is_approved=1")
        total_slp = int(cur.fetchone()[0] or 0)

        cur.execute("SELECT COUNT(*) FROM records_analysis")
        total_practice = int(cur.fetchone()[0] or 0)

        cur.execute("SELECT COUNT(*) FROM service_sessions")
        total_sessions = int(cur.fetchone()[0] or 0)

        # -------------------------
        # Series (by day in range)
        # -------------------------
        days = _date_series(d1, d2)
        base = {d: 0 for d in days}

        # new patients per day
        cur.execute("""
            SELECT date(created_at) AS d, COUNT(*) AS c
            FROM patients
            WHERE date(created_at) BETWEEN date(?) AND date(?)
            GROUP BY date(created_at)
        """, (d1, d2))
        new_patients_map = dict(base)
        for r in cur.fetchall():
            new_patients_map[str(r[0])] = int(r[1] or 0)

        # new slp per day (approved or all? use created_at regardless, but keep role='slp')
        cur.execute("""
            SELECT date(created_at) AS d, COUNT(*) AS c
            FROM staff_users
            WHERE role='slp'
              AND date(created_at) BETWEEN date(?) AND date(?)
            GROUP BY date(created_at)
        """, (d1, d2))
        new_slps_map = dict(base)
        for r in cur.fetchall():
            new_slps_map[str(r[0])] = int(r[1] or 0)

        # sessions created per day
        cur.execute("""
            SELECT date(created_at) AS d, COUNT(*) AS c
            FROM service_sessions
            WHERE date(created_at) BETWEEN date(?) AND date(?)
            GROUP BY date(created_at)
        """, (d1, d2))
        sess_created_map = dict(base)
        for r in cur.fetchall():
            sess_created_map[str(r[0])] = int(r[1] or 0)

        # sessions closed per day
        cur.execute("""
            SELECT date(closed_at) AS d, COUNT(*) AS c
            FROM service_sessions
            WHERE status='closed'
              AND closed_at IS NOT NULL
              AND date(closed_at) BETWEEN date(?) AND date(?)
            GROUP BY date(closed_at)
        """, (d1, d2))
        sess_closed_map = dict(base)
        for r in cur.fetchall():
            sess_closed_map[str(r[0])] = int(r[1] or 0)

        # practice records per day
        cur.execute("""
            SELECT date(created_at) AS d, COUNT(*) AS c
            FROM records_analysis
            WHERE date(created_at) BETWEEN date(?) AND date(?)
            GROUP BY date(created_at)
        """, (d1, d2))
        practice_map = dict(base)
        for r in cur.fetchall():
            practice_map[str(r[0])] = int(r[1] or 0)

        def to_series(m: Dict[str, int]) -> List[Dict[str, Any]]:
            return [{"date": d, "value": int(m.get(d, 0))} for d in days]

        def to_series2(m1: Dict[str, int], m2: Dict[str, int], k1: str, k2: str) -> List[Dict[str, Any]]:
            out = []
            for d in days:
                out.append({"date": d, k1: int(m1.get(d, 0)), k2: int(m2.get(d, 0))})
            return out

        return {
            "range": {"from": d1, "to": d2},
            "totals": {
                "patients": total_patients,
                "slps": total_slp,
                "service_sessions": total_sessions,
                "practice_records": total_practice
            },
            "series": {
                "new_patients_daily": to_series(new_patients_map),
                "new_slps_daily": to_series(new_slps_map),
                "sessions_daily": to_series2(sess_created_map, sess_closed_map, "created", "closed"),
                "practice_daily": to_series(practice_map),
            }
        }
    finally:
        conn.close()



# ------------------------------
# English practice bank helpers
# ------------------------------
EN_BANK_FILES = {
    "Addition": "LLM/Addition.json",
    "Omission": "LLM/Omission.json",
    "Substitution": "LLM/Substitution.json",
}

def infer_en_difficulty_from_user_id(user_id: str) -> str:
    """
    Return primary|secondary|advanced from patients.age
    Same policy as /api/patient/level
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT age FROM patients WHERE id=?", (user_id,))
    row = cur.fetchone()
    conn.close()

    age = row[0] if row else None
    try:
        age_int = int(age) if age is not None else None
    except Exception:
        age_int = None

    if age_int is None:
        return "primary"
    if age_int < 12:
        return "primary"
    if age_int < 18:
        return "secondary"
    return "advanced"

def normalize_difficulty_bucket_en(level: str | None) -> str:
    """
    English bank 只接受: primary|secondary|advanced
    允許前端傳: primary_school / 小學 / 中學 / 大學 等
    """
    s = (level or "").strip().lower()
    mapping = {
        "primary_school": "primary",
        "primary": "primary",
        "elementary": "primary",
        "小學": "primary",
        "小学": "primary",

        "secondary": "secondary",
        "middle": "secondary",
        "中學": "secondary",
        "中学": "secondary",

        "advanced": "advanced",
        "advance": "advanced",
        "大學": "advanced",
        "大学": "advanced",
    }
    return mapping.get(s, "advanced")

def load_en_bank_items(super_type: str, sub_type: str, difficulty: str) -> list[dict]:
    """
    從 LLM/*.json (NEW format) 讀取指定 (super_type, sub_type, difficulty) 的 items。
    回傳格式: [{"w": "...", "ipa": "/.../"}, ...]
    """
    super_type = (super_type or "").strip()
    sub_type = (sub_type or "").strip()
    difficulty = normalize_difficulty_bucket_en(difficulty)

    path = EN_BANK_FILES.get(super_type)
    if not path:
        return []

    obj = safe_load_json(path)  # 你 main.py 已經有 safe_load_json
    banks = obj.get("banks")
    if not isinstance(banks, dict):
        return []

    sub = banks.get(sub_type)
    if not isinstance(sub, dict):
        return []

    lvl = sub.get(difficulty)
    if not isinstance(lvl, dict):
        return []

    items = lvl.get("items", [])
    if not isinstance(items, list):
        return []

    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        w = it.get("w")
        if not isinstance(w, str) or not w.strip():
            continue
        out.append({
            "w": w.strip(),
            "ipa": str(it.get("ipa") or "").strip()
        })
    return out

class NextWordEnResp(BaseModel):
    super_type: str
    sub_type: str
    language: str
    difficulty: str
    index: int          # 1-based
    total_items: int
    target_word: str
    target_ipa: str
    record_id: str

@app.get("/api/practice/en/next-word", response_model=NextWordEnResp)
def api_practice_next_word_en(
    super_type: str = Query(..., description="Addition|Omission|Substitution"),
    sub_type: str = Query(..., description="Key in LLM/<super>.json banks"),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    English take vocab:
    - difficulty 由後端依 current_user_id -> patients.age 自動推斷
    - 根據 difficulty 從 LLM/*.json 的 banks[sub_type][difficulty].items 取詞
    """
    ensure_practice_progress_table()

    super_type = (super_type or "").strip()
    sub_type = (sub_type or "").strip()

    if super_type not in EN_BANK_FILES:
        raise HTTPException(status_code=400, detail=f"Unsupported super_type: {super_type}")
    if not sub_type:
        raise HTTPException(status_code=400, detail="Missing sub_type")

    # ✅ 後端自動推斷 difficulty
    diff = infer_en_difficulty_from_user_id(current_user_id)  # primary|secondary|advanced

    items = load_en_bank_items(super_type, sub_type, diff)
    total_items = len(items)
    if total_items <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"Empty bank: {super_type}/{sub_type} difficulty={diff}"
        )

    lang = "en"

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, next_start_index, chunk_size
        FROM practice_progress
        WHERE user_id=? AND super_type=? AND sub_type=? AND language=?
    """, (current_user_id, super_type, sub_type, lang))
    row = cur.fetchone()

    if row:
        prog_id, next_idx, chunk_size = row[0], int(row[1] or 0), int(row[2] or 50)
    else:
        prog_id = str(uuid.uuid4())
        next_idx = 0
        chunk_size = 50
        cur.execute("""
            INSERT INTO practice_progress
            (id, user_id, super_type, sub_type, language, chunk_size, next_start_index, last_range_end)
            VALUES (?, ?, ?, ?, ?, ?, 0, 0)
        """, (prog_id, current_user_id, super_type, sub_type, lang, chunk_size))
        conn.commit()

    if next_idx >= total_items:
        next_idx = 0
        cur.execute("""
            UPDATE practice_progress
            SET next_start_index=0, last_range_end=0, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (prog_id,))
        conn.commit()

    pick = items[next_idx]
    target_word = pick["w"]
    target_ipa = (pick.get("ipa") or "").replace("/", "").strip()

    record_id = str(uuid.uuid4())

    conn.close()

    return NextWordEnResp(
        super_type=super_type,
        sub_type=sub_type,
        language="en",
        difficulty=diff,
        index=next_idx + 1,
        total_items=total_items,
        target_word=target_word,
        target_ipa=target_ipa,
        record_id=record_id
    )


@app.get("/api/practice/en/subtypes")
def api_practice_en_subtypes(
    super_type: str = Query(...),
):
    super_type = (super_type or "").strip()
    path = EN_BANK_FILES.get(super_type)
    if not path:
        raise HTTPException(status_code=400, detail="Unsupported super_type")

    obj = safe_load_json(path)
    banks = obj.get("banks") or {}
    if not isinstance(banks, dict):
        return {"super_type": super_type, "subtypes": []}

    return {"super_type": super_type, "subtypes": sorted(list(banks.keys()))}
    

#-----Chinese Get Word ---------------
@app.get("/api/practice/next-words", response_model=NextWordsResp)
def api_practice_next_words(
    super_type: str = Query(...),
    sub_type: str = Query(...),
    language: str = Query("en"),
    chunk_size: int = Query(50),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    用 practice_progress 去 slpErrors / bank 取下一段詞
    - 不會自動 commit；前端做完一段再 call /api/practice/commit-range
    """
    ensure_practice_progress_table()

    super_type = (super_type or "").strip()
    sub_type = (sub_type or "").strip()
    lang = normalize_practice_language(language)
    chunk_size = int(chunk_size or 50)

    if not super_type or not sub_type:
        raise HTTPException(status_code=400, detail="Missing super_type/sub_type")
    if chunk_size <= 0 or chunk_size > 200:
        raise HTTPException(status_code=400, detail="Invalid chunk_size")

    # ✅ 目前你的詞庫是 slpErrors（flatten 後 dict）
    if sub_type not in slpErrors:
        raise HTTPException(status_code=400, detail=f"Unknown sub_type: {sub_type}")

    bank = slpErrors[sub_type]
    total_items = len(bank)
    if total_items == 0:
        raise HTTPException(status_code=400, detail="Empty bank")

    # 讀 progress
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT next_start_index, chunk_size
        FROM practice_progress
        WHERE user_id=? AND super_type=? AND sub_type=? AND language=?
    """, (current_user_id, super_type, sub_type, lang))
    row = cur.fetchone()
    conn.close()

    if row:
        next_start_index = int(row[0] or 0)
        # 你可以選擇「用 DB chunk_size」或「用 query chunk_size」
        # 這裡採用 query chunk_size（前端可控制）
    else:
        next_start_index = 0

    start = next_start_index
    end_excl = min(start + chunk_size, total_items)

    # 已做完
    if start >= total_items:
        return NextWordsResp(
            super_type=super_type,
            sub_type=sub_type,
            language=lang,
            range_start=total_items + 1,
            range_end=total_items,
            total_items=total_items,
            items=[]
        )

    items = bank[start:end_excl]

    return NextWordsResp(
        super_type=super_type,
        sub_type=sub_type,
        language=lang,
        range_start=start + 1,
        range_end=end_excl,
        total_items=total_items,
        items=items
    )

def migrate_practice_progress_unique_with_language():
    """
    SQLite migration:
    - Ensure practice_progress has language column
    - Rebuild table with UNIQUE(user_id, super_type, sub_type, language)
    - Preserve existing data (old rows default language='en')
    """
    conn = get_db()
    cur = conn.cursor()

    try:
        # 0) 確認舊表存在
        cur.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='practice_progress'
        """)
        if not cur.fetchone():
            # 沒表就直接建新表（按你新 schema）
            cur.execute("""
            CREATE TABLE practice_progress (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                super_type TEXT NOT NULL,
                sub_type TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'en',
                chunk_size INTEGER NOT NULL DEFAULT 50,
                next_start_index INTEGER NOT NULL DEFAULT 0,
                last_range_end INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES patients(id) ON DELETE CASCADE,
                UNIQUE(user_id, super_type, sub_type, language)
            )
            """)
            conn.commit()
            print("🧿 practice_progress table created (new schema).")
            return

        # 1) 檢查舊表有冇 language 欄（如果冇，搬資料時視作 'en'）
        cur.execute("PRAGMA table_info(practice_progress);")
        cols = [r[1] for r in cur.fetchall()]
        has_language = "language" in cols

        # 2) Rename 舊表
        cur.execute("ALTER TABLE practice_progress RENAME TO practice_progress_old;")

        # 3) 建新表（含 language + 新 UNIQUE）
        cur.execute("""
        CREATE TABLE practice_progress (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            super_type TEXT NOT NULL,
            sub_type TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'en',
            chunk_size INTEGER NOT NULL DEFAULT 50,
            next_start_index INTEGER NOT NULL DEFAULT 0,
            last_range_end INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES patients(id) ON DELETE CASCADE,
            UNIQUE(user_id, super_type, sub_type, language)
        )
        """)

        # 4) 搬資料
        # 如果舊表冇 language，就全部當 'en'
        if has_language:
            cur.execute("""
                INSERT INTO practice_progress
                (id, user_id, super_type, sub_type, language, chunk_size, next_start_index, last_range_end, updated_at)
                SELECT
                  id, user_id, super_type, sub_type,
                  COALESCE(NULLIF(TRIM(language), ''), 'en') AS language,
                  chunk_size, next_start_index, last_range_end,
                  COALESCE(updated_at, CURRENT_TIMESTAMP)
                FROM practice_progress_old
            """)
        else:
            cur.execute("""
                INSERT INTO practice_progress
                (id, user_id, super_type, sub_type, language, chunk_size, next_start_index, last_range_end, updated_at)
                SELECT
                  id, user_id, super_type, sub_type,
                  'en' AS language,
                  chunk_size, next_start_index, last_range_end,
                  COALESCE(updated_at, CURRENT_TIMESTAMP)
                FROM practice_progress_old
            """)

        # 5) 可選：重建常用 index（你 create_db.py 有，但 main.py 未必需要）
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_practice_progress_user_bank_lang
            ON practice_progress(user_id, super_type, sub_type, language);
        """)

        # 6) Drop 舊表
        cur.execute("DROP TABLE practice_progress_old;")

        conn.commit()
        print("✅ practice_progress migrated: UNIQUE(user_id, super_type, sub_type, language)")

    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()

class NextWordResp(BaseModel):
    super_type: str
    sub_type: str
    language: str
    index: int          # 1-based
    total_items: int
    word: str
    record_id: str      # 用於把「讀的結果」寫回 DB（optional 但建議）

def get_practice_bank_by_super_type(super_type: str) -> dict:
    st = (super_type or "").strip()

    if st == "Substitution":
        # ✅ 你想用邊份 substitution bank，就用邊個：
        # 1) 如果你要用 LLM1/substitution.json：
        return substitution_errors_llm1
        # 2) 如果你要用 LLM/Substitution.json (banks 格式轉完嘅 substitution_errors)：
        # return substitution_errors

    if st == "InitialConsonant":
        return InitialConsonant

    if st == "Omission":
        return omission_errors  # 或 omission_errors_llm1（你要統一一份）

    if st == "Addition":
        return ADDITION_BANK

    if st == "Tone":
        return Tone_errors_llm1

    if st == "Distortion":
        return distortion_errors_llm1

    if st == "VowelFinal":
        return VowelFinal

    # fallback（可留可唔留）
    return slpErrors

#-------English get word ----------------------------------------
@app.get("/api/practice/next-word", response_model=NextWordResp)
def api_practice_next_word(
    super_type: str = Query(...),
    sub_type: str = Query(...),
    language: str = Query("en"),
    current_user_id: str = Depends(get_current_user_id),
    difficulty_level: str = Query("小學"),
):
    ensure_practice_progress_table()

    lang = normalize_practice_language(language)

    bank_map = get_practice_bank_by_super_type(super_type)

    if sub_type not in bank_map:
        raise HTTPException(status_code=400, detail=f"Unknown sub_type: {sub_type} (super_type={super_type})")

    bank = bank_map[sub_type]
        
    total_items = len(bank)
    if total_items <= 0:
        raise HTTPException(status_code=400, detail="Empty bank")

    conn = get_db()
    cur = conn.cursor()

    # 1) 讀 progress（冇就建一行）
    cur.execute("""
        SELECT id, next_start_index, chunk_size
        FROM practice_progress
        WHERE user_id=? AND super_type=? AND sub_type=? AND language=?
    """, (current_user_id, super_type, sub_type, lang))
    row = cur.fetchone()

    if row:
        prog_id, next_idx, chunk_size = row[0], int(row[1] or 0), int(row[2] or 50)
    else:
        prog_id = str(uuid.uuid4())
        next_idx = 0
        chunk_size = 50
        cur.execute("""
            INSERT INTO practice_progress
            (id, user_id, super_type, sub_type, language, chunk_size, next_start_index, last_range_end)
            VALUES (?, ?, ?, ?, ?, ?, 0, 0)
        """, (prog_id, current_user_id, super_type, sub_type, lang, chunk_size))
        conn.commit()

    # 2) 若做完全部 → 重置回 0（再回到 1-50）
    if next_idx >= total_items:
        next_idx = 0
        cur.execute("""
            UPDATE practice_progress
            SET next_start_index=0, last_range_end=0, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (prog_id,))
        conn.commit()

    # 3) 取下一個詞
    word = bank[next_idx]

    # 4) 建立一筆 records_analysis 記錄「派題」
    record_id = save_word_take_record(
        user_id=current_user_id,
        language="Chinese" if lang == "cn" else "English",  # 你的 records_analysis.language 目前是存 "Chinese"/"English"
        test_type=super_type,
        category=sub_type,              # 或 pattern；但對 practice bank 先用 sub_type 較合理
        target_word=word,
        difficulty_level=normalize_difficulty(difficulty_level),
    )

    conn.close()

    return NextWordResp(
        super_type=super_type,
        sub_type=sub_type,
        language=lang,
        index=next_idx + 1,
        total_items=total_items,
        word=word,
        record_id=record_id
    )

class CommitOneReq(BaseModel):
    super_type: str
    sub_type: str
    language: str = "en"
    record_id: str

    asr_word: str = ""
    asr_ipa: str = ""

    # ✅ new: score (front-end send back)
    score: Optional[int] = None

    correct_rate: Optional[int] = None
    check_results: dict = {}

    task_id: Optional[str] = None
    task_kind: str = "vocab"
    task_increment: int = 1

@app.post("/api/practice/commit-one")
def api_practice_commit_one(
    data: CommitOneReq,
    current_user_id: str = Depends(get_current_user_id),
):
    ensure_practice_progress_table()
    ensure_task_table_v2()  # ✅ ensure tasks columns exist

    super_type = (data.super_type or "").strip()
    sub_type = (data.sub_type or "").strip()
    lang = normalize_practice_language(data.language)

    if not super_type or not sub_type:
        raise HTTPException(status_code=400, detail="Missing super_type/sub_type")

    task_kind = (data.task_kind or "vocab").strip().lower()
    if task_kind not in ("vocab", "sentence"):
        raise HTTPException(status_code=400, detail="Invalid task_kind (vocab|sentence)")

    inc = int(data.task_increment or 0)
    if inc < 0 or inc > 10:
        raise HTTPException(status_code=400, detail="Invalid task_increment")

    conn = get_db()
    cur = conn.cursor()

    try:
        # -------------------------
        # 0) verify record ownership
        # -------------------------
        cur.execute("SELECT user_id FROM records_analysis WHERE id=?", (data.record_id,))
        r = cur.fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="record_id not found")
        if r[0] != current_user_id:
            raise HTTPException(status_code=403, detail="not allowed")

        # -------------------------
        # 1) TASK gate + increment (if task_id provided & inc>0)
        # today-only + day_index gating + prev day must completed
        # -------------------------
        if data.task_id and inc > 0:
            cur.execute("""
                SELECT
                  id, user_id, super_type, sub_type, task_kind, language,
                  allocated_at, day_index, valid_days,
                  total_times, progress_times, status, is_active
                FROM tasks
                WHERE id=?
                LIMIT 1
            """, (data.task_id,))
            t = cur.fetchone()
            if not t:
                raise HTTPException(status_code=404, detail="task_id not found")

            (tid, uid, t_super, t_sub, t_kind, t_lang,
             allocated_at, day_index, valid_days,
             total_times, progress_times, status, is_active) = t

            if uid != current_user_id:
                raise HTTPException(status_code=403, detail="task not allowed")

            if int(is_active or 0) != 1:
                raise HTTPException(status_code=400, detail="task inactive")

            # must match this practice
            if str(t_super) != super_type or str(t_sub) != sub_type:
                raise HTTPException(status_code=400, detail="task super_type/sub_type mismatch")
            if normalize_practice_language(t_lang) != lang:
                raise HTTPException(status_code=400, detail="task language mismatch")
            if str(t_kind).lower() != task_kind:
                raise HTTPException(status_code=400, detail="task_kind mismatch")

            # ✅ today-only: allowed day index must equal day_index
            allowed = _task_allowed_day_index(allocated_at)
            if allowed is None:
                raise HTTPException(status_code=400, detail="task expired or invalid allocated_at")

            if int(day_index) != int(allowed):
                # today only can do today task; cannot do past/future
                raise HTTPException(status_code=400, detail=f"NOT_TODAY_TASK_DAY allowed_day_index={allowed}")

            # increment task progress
            total = int(total_times or 0)
            prog = int(progress_times or 0)
            new_prog = prog + inc
            if total > 0:
                new_prog = min(new_prog, total)

            new_status = status
            if total > 0 and new_prog >= total:
                new_status = "completed"

            cur.execute("""
                UPDATE tasks
                SET progress_times=?,
                    status=?
                WHERE id=?
            """, (new_prog, new_status, tid))

        # -------------------------
        # 2-0) ✅ prevent double commit points
        # -------------------------
        cur.execute("SELECT check_results FROM records_analysis WHERE id=?", (data.record_id,))
        row = cur.fetchone()
        old_check_results = {}
        if row and row[0]:
            try:
                old_check_results = json.loads(row[0])
            except Exception:
                old_check_results = {}

        already_committed = isinstance(old_check_results, dict) and old_check_results.get("event") == "read_word"

        # -------------------------
        # 2) update records_analysis
        # -------------------------
        cur.execute("""
            UPDATE records_analysis
            SET asr_word=?,
                asr_ipa=?,
                correct_rate=?,
                check_results=?,
                created_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (
            data.asr_word,
            data.asr_ipa,
            str(data.correct_rate) if data.correct_rate is not None else None,
            json.dumps({"event": "read_word", **(data.check_results or {})}, ensure_ascii=False),
            data.record_id
        ))

        # -------------------------
        # 3) push practice_progress +1
        # -------------------------
        cur.execute("""
            UPDATE practice_progress
            SET next_start_index = next_start_index + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id=? AND super_type=? AND sub_type=? AND language=?
        """, (current_user_id, super_type, sub_type, lang))

        if cur.rowcount != 1:
            raise HTTPException(status_code=400, detail="practice_progress row not found (call next-word first)")
        
        # -------------------------
        # 4) ✅ add integral (score) for patient header
        # -------------------------
        try:
            pts = int(data.score) if data.score is not None else 0
        except Exception:
            pts = 0

        if (not already_committed) and pts > 0:
            cur.execute(
                "UPDATE patients SET integral = COALESCE(integral,0) + ? WHERE id=?",
                (pts, current_user_id)
            )

        conn.commit()
        return {"msg": "ok", "record_id": data.record_id}

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"commit-one failed: {str(e)}")
    finally:
        conn.close()

def save_word_take_record(
    user_id: str,
    language: str,
    test_type: str,
    category: str,
    target_word: str,
    difficulty_level: str = "primary",
):
    return save_records_analysis_v2(
        user_id=user_id,
        language=language,
        test_type=test_type,
        category=category,
        target_word=target_word,
        target_ipa="",
        asr_ipa="",
        asr_word="",
        correct_rate=None,
        check_results={"event": "take_word"},
        difficulty_level=difficulty_level,
        score=None,
        severity=None,
    )

from fastapi import UploadFile, File, Form

@app.post("/api/practice/commit-one-with-audio")
async def api_practice_commit_one_with_audio(
    record_id: str = Form(...),
    super_type: str = Form(...),
    sub_type: str = Form(...),
    language: str = Form("cn"),
    asr_word: str = Form(""),
    asr_ipa: str = Form(""),
    correct_rate: str = Form(""),
    check_results: str = Form("{}"),
    score: str = Form(""),
    file: UploadFile = File(...),
    current_user_id: str = Depends(get_current_user_id),
):
    ensure_practice_progress_table()

    # 0) verify record ownership
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM records_analysis WHERE id=?", (record_id,))
    r = cur.fetchone()
    if not r:
        conn.close()
        raise HTTPException(status_code=404, detail="record_id not found")
    if r[0] != current_user_id:
        conn.close()
        raise HTTPException(status_code=403, detail="not allowed")

    # 1) save audio
    os.makedirs(UPLOAD_AUDIO_DIR, exist_ok=True)
    ext = ".wav"
    fn = f"{record_id}{ext}"
    save_path = os.path.join(UPLOAD_AUDIO_DIR, fn)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    audio_url = f"/uploads/audio/{fn}"

    # 2) parse json
    try:
        check_obj = json.loads(check_results) if check_results else {}
    except Exception:
        check_obj = {"raw": check_results}

    # 3) normalize numeric
    cr = None
    try:
        cr = int(float(correct_rate)) if str(correct_rate).strip() != "" else None
    except Exception:
        cr = None

    sc = None
    try:
        sc = int(float(score)) if str(score).strip() != "" else None
    except Exception:
        sc = None

    # fallback scoring if not provided
    if sc is None and cr is not None:
        sc = int(cr)  # 最簡單：score=correct_rate（你之後可改規則）

    # 4) update records_analysis (store audio path + score)
    cur.execute("""
        UPDATE records_analysis
        SET asr_word=?,
            asr_ipa=?,
            correct_rate=?,
            score=?,
            check_results=?,
            created_at=CURRENT_TIMESTAMP
        WHERE id=?
    """, (
        asr_word,
        asr_ipa,
        str(cr) if cr is not None else None,
        sc,
        json.dumps({"event": "read_word", "audio_url": audio_url, **(check_obj or {})}, ensure_ascii=False),
        record_id
    ))

    # 5) push progress
    lang = normalize_practice_language(language)
    cur.execute("""
        UPDATE practice_progress
        SET next_start_index = next_start_index + 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id=? AND super_type=? AND sub_type=? AND language=?
    """, (current_user_id, super_type, sub_type, lang))

    if cur.rowcount != 1:
        conn.rollback()
        conn.close()
        raise HTTPException(status_code=400, detail="practice_progress row not found (call next-word first)")

    # 6) add integral (score)
    if sc:
        cur.execute("UPDATE patients SET integral = COALESCE(integral,0) + ? WHERE id=?", (sc, current_user_id))

    conn.commit()
    conn.close()


    return {"msg": "ok", "record_id": record_id, "audio_url": audio_url, "score": sc}

def award_points_on_task_completed(cur, user_id: str, task_id: str, points: int = 100):
    """
    當 task 第一次完成（pending -> completed）時，加分一次。
    依賴 tasks.status 在同一個 transaction 內正確更新。
    """
    points = int(points or 0)
    if points <= 0:
        return

    # ✅ 加分
    cur.execute(
        "UPDATE patients SET integral = COALESCE(integral, 0) + ? WHERE id=?",
        (points, user_id)
    )

def patient_friendly_explain(check_results: dict) -> dict:
    """
    Convert technical detector outputs into patient-friendly English explanations.

    Return format:
      {
        "summary": "...",
        "items": {
          "substitution": {"explain": "...", "tip": "..."},
          ...
        }
      }
    """
    out = {"summary": "", "items": {}}

    if not isinstance(check_results, dict):
        return {
            "summary": "No analysis details available.",
            "items": {}
        }

    meta = check_results.get("meta", {}) if isinstance(check_results.get("meta", {}), dict) else {}
    off = meta.get("off_target", {}) if isinstance(meta.get("off_target", {}), dict) else {}

    # Off-target first (high priority)
    if isinstance(off, dict) and off.get("is_off_target"):
        out["summary"] = (
            "The system may have recognized your speech as a different word (off‑target). "
            "Try speaking more slowly and clearly syllable by syllable, then try again."
        )

    def add_item(k: str, explain: str, tip: str):
        out["items"][k] = {"explain": explain, "tip": tip}

    # ---- substitution ----
    sub = check_results.get("substitution")
    if isinstance(sub, dict) and sub.get("has_substitution"):
        typ = sub.get("type")
        if typ == "声母替代":
            add_item(
                "substitution",
                "Your initial consonant (the beginning sound) seems to have changed to a different one (e.g., z/zh, c/ch, s/sh).",
                "Exaggerate the initial consonant first, then add the final. Practice: initial → final → full syllable."
            )
        elif typ == "韵母替代":
            add_item(
                "substitution",
                "Your final/vowel part seems to have changed to a different final.",
                "Keep a stable mouth shape (opening, lip rounding, tongue position). Practice the final alone, then combine it with the initial."
            )
        elif typ == "声调替代":
            add_item(
                "substitution",
                "Your tone pattern does not match the target word.",
                "Hum the tone contour first (Tone 1 level, Tone 2 rising, Tone 3 dipping, Tone 4 falling), then apply it to the syllable."
            )
        else:
            add_item(
                "substitution",
                "A substitution difference was detected between your pronunciation and the target.",
                "Slow down and read syllable by syllable. Check initial consonant, final, and tone."
            )

    # ---- tone ----
    tone = check_results.get("tone")
    if isinstance(tone, dict) and tone.get("has_tone_error"):
        typ = tone.get("type")
        if typ == "声调替代":
            add_item(
                "tone",
                "The tone (tone number) seems incorrect.",
                "Practice the tone contour alone, then read the word."
            )
        elif typ == "声调错位":
            add_item(
                "tone",
                "This may involve tone sandhi (tone change rules), such as consecutive third tones or the word '一'.",
                "Apply tone sandhi rules slowly (e.g., 3+3 → 2+3). Then speed up gradually."
            )
        elif typ == "轻声错误":
            add_item(
                "tone",
                "The target includes a neutral tone, but you pronounced it with a full tone.",
                "Neutral tone should be lighter and shorter; place the stress on the previous syllable."
            )
        else:
            add_item(
                "tone",
                "A tone-related issue was detected.",
                "Confirm each syllable’s tone first, then read the word smoothly."
            )

    # ---- vowel_final ----
    vf = check_results.get("vowel_final")
    if isinstance(vf, dict) and vf.get("has_vowel_error"):
        typ = vf.get("type")
        if typ == "单元音替代":
            add_item(
                "vowel_final",
                "A simple vowel (a/o/e/i/u/ü) may have been produced differently than expected.",
                "Use a mirror to stabilize mouth opening, lip rounding, and tongue position."
            )
        elif typ == "复合韵母错误":
            add_item(
                "vowel_final",
                "A compound final (e.g., ai/ao/ou/ang/eng) may be incomplete or in the wrong sequence.",
                "Break the final into parts and say it slowly, then blend it smoothly."
            )
        else:
            add_item(
                "vowel_final",
                "The final (vowel part) does not match the target.",
                "Practice the final first, then add the initial consonant."
            )

    # ---- initial (initial consonant category) ----
    ini = check_results.get("initial")
    if isinstance(ini, dict) and ini.get("has_initial_error"):
        typ = ini.get("type")
        if typ == "声母省略":
            add_item(
                "initial",
                "The initial consonant may have been omitted (it sounds like the syllable starts directly with a vowel).",
                "Practice producing the initial consonant briefly and clearly, then attach the final."
            )
        elif typ == "声母添加":
            add_item(
                "initial",
                "An extra initial consonant may have been added (the target might start without that consonant).",
                "Confirm whether the target syllable needs an initial. Try starting from the final and then adding the correct initial."
            )
        elif typ == "发音部位错误":
            add_item(
                "initial",
                "The place of articulation may be incorrect (e.g., tongue position for retroflex vs. non‑retroflex).",
                "Practice tongue placement and airflow. Compare pairs like z vs zh, c vs ch, s vs sh."
            )
        else:  # 声母替代
            add_item(
                "initial",
                "The initial consonant seems to have changed to a different consonant.",
                "Slow down and focus on contrasts such as aspiration and retroflex vs. non‑retroflex."
            )

    # ---- omission ----
    omi = check_results.get("omission")
    if isinstance(omi, dict) and omi.get("has_omission"):
        typ = omi.get("type")
        if typ == "声母脱落":
            add_item(
                "omission",
                "The initial consonant may be missing.",
                "Slow down and make the initial consonant clear (especially b/p/m, d/t, zh/ch/sh)."
            )
        elif typ == "韵尾脱落":
            add_item(
                "omission",
                "The ending nasal sound (n/ng) may be missing.",
                "Close the ending properly: n uses the tongue tip; ng uses the back of the tongue."
            )
        elif typ == "韵母脱落":
            add_item(
                "omission",
                "The vowel/final may be shortened or partially missing.",
                "Hold the vowel long enough, then add the tone."
            )
        elif typ == "声调脱落":
            add_item(
                "omission",
                "The tone marking may be missing (it sounds like a neutral/untone syllable).",
                "Make sure each syllable carries a tone contour; start slow, then increase speed."
            )
        else:
            add_item(
                "omission",
                "A possible omission was detected.",
                "Read syllable by syllable to ensure each part is fully produced."
            )

    # ---- addition ----
    add = check_results.get("addition")
    if isinstance(add, dict) and add.get("has_addition"):
        typ = add.get("type")
        if typ == "声母添加":
            add_item(
                "addition",
                "An extra initial consonant may have been added.",
                "Avoid extra bursts/friction at the start; enter the correct initial directly."
            )
        elif typ == "韵母添加":
            add_item(
                "addition",
                "An extra vowel/final element may have been added (often from over‑lengthening).",
                "Keep the syllable clean and concise; avoid dragging the vowel."
            )
        elif typ == "韵尾添加":
            add_item(
                "addition",
                "An extra ending (n/ng) may have been added.",
                "Shorten the ending and avoid adding nasal sounds."
            )
        else:  # 插音
            add_item(
                "addition",
                "An extra transitional sound may have been inserted between syllables.",
                "Slow down and avoid adding a 'bridge sound' when connecting syllables."
            )

    # ---- distortion ----
    dis = check_results.get("distortion")
    if isinstance(dis, dict) and dis.get("has_distortion"):
        add_item(
            "distortion",
            "This word is in the distortion training set and may require extra attention to sound quality.",
            "Practice mouth shape and airflow for the target sound class; start slow and speed up gradually."
        )

    # Default summary if empty
    if not out["summary"]:
        has_any_error = any(
            isinstance(v, dict) and any(v.get(k) for k in (
                "has_addition",
                "has_substitution",
                "has_tone_error",
                "has_distortion",
                "has_omission",
                "has_initial_error",
                "has_vowel_error",
            ))
            for v in check_results.values()
        )

        out["summary"] = (
            "A difference from the target pronunciation was detected. Try slowing down and practicing syllable by syllable."
            if has_any_error
            else "Your pronunciation looks correct overall. Keep the same clarity and pacing."
        )

    return out

TEST_LANG = "cn"  # fixed for this screening test

TEST_BANKS = {
    "Addition": safe_load_json("LLM1/Addition.json"),
    "Distortion": safe_load_json("LLM1/Distortion.json"),
    "Omission": safe_load_json("LLM1/omission.json"),
    "Substitution": safe_load_json("LLM1/substitution.json"),
    "Tone": safe_load_json("LLM1/Tone.json"),
    "VowelFinal": safe_load_json("LLM1/VowelFinal.json"),
    "InitialConsonant": safe_load_json("LLM1/InitialConsonant.json"),
}

def create_test_session_row_v2(
    user_id: str,
    test_name: str,
    *,
    language: str,
    words_per_subtype: int,
    threshold: int,
    plan_obj: dict,
    total_questions: int,
    config_obj: dict | None = None,
) -> str:
    """
    建立 tests(super test) 一筆資料（新 schema）
    - 必須寫入 plan（因為 tests.plan NOT NULL）
    - 寫入 total_questions/current_question_index，方便 resume
    """
    conn = get_db()
    cur = conn.cursor()
    tid = str(uuid.uuid4())

    cur.execute("""
        INSERT INTO tests (
            id, patient_id,
            test_name, status,
            progress_cursor, total_questions, current_question_index,
            plan, config_json,
            threshold, words_per_subtype, language,
            has_slp, result,
            date, updated_at, finished_at
        ) VALUES (
            ?, ?,
            ?, 'running',

            0, ?, 0,

            ?, ?,

            ?, ?, ?,

            0, NULL,

            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL
        )
    """, (
        tid, user_id,
        test_name,
        int(total_questions),
        json.dumps(plan_obj, ensure_ascii=False),
        json.dumps(config_obj or {}, ensure_ascii=False),
        int(threshold),
        int(words_per_subtype),
        str(language),
    ))
    conn.commit()
    conn.close()
    return tid


def update_test_session_result(test_id: str, has_slp: int, result_obj: dict):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        UPDATE tests
        SET has_slp=?,
            result=?,
            status='finished',
            updated_at=CURRENT_TIMESTAMP,
            finished_at=CURRENT_TIMESTAMP
        WHERE id=?
    """, (int(has_slp or 0), json.dumps(result_obj, ensure_ascii=False), test_id))
    conn.commit()
    conn.close()


def insert_slp_patient_rows(
    user_id: str,
    test_id: str,
    confirmed: list[tuple[str, str]],  # [(super_type, sub_type), ...]
    language: str = TEST_LANG,
):
    """
    Insert rows into slp_patient for confirmed SLP categories.
    (No de-dup logic enforced by schema, so we do safe dedup in code.)
    """
    if not confirmed:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    uniq = []
    seen = set()
    for st, sb in confirmed:
        key = (str(st), str(sb))
        if key not in seen:
            seen.add(key)
            uniq.append(key)

    conn = get_db()
    cur = conn.cursor()

    for st, sb in uniq:
        cur.execute("""
            INSERT INTO slp_patient (id, user_id, test_id, datetime, super_type, sub_type, language)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), user_id, test_id, now, st, sb, language))

    conn.commit()
    conn.close()

from pydantic import BaseModel
from typing import List, Dict, Any

TEST_SUPER_ORDER = [
    "Addition",
    "Distortion",
    "Omission",
    "Substitution",
    "Tone",
    "VowelFinal",
    "InitialConsonant",
]

class TestStartResp(BaseModel):
    test_id: str
    language: str
    words_per_subtype: int
    plan: List[Dict[str, Any]]  # [{super_type, sub_type, words:[...]}]
    total_questions: int


def _pick_n_words(bank_list: list[str], n: int) -> list[str]:
    if not bank_list:
        return []
    if len(bank_list) <= n:
        # shuffle copy
        tmp = list(bank_list)
        random.shuffle(tmp)
        return tmp
    # sample unique
    return random.sample(bank_list, n)


@app.post("/api/test/start", response_model=TestStartResp)
def api_test_start(
    words_per_subtype: int = Query(None, description="optional override; backend default if omitted"),
    current_user_id: str = Depends(get_current_user_id),
):
    n = int(words_per_subtype or TEST_WORDS_PER_SUBTYPE)
    if n <= 0 or n > 20:
        raise HTTPException(status_code=400, detail="Invalid words_per_subtype (1..20)")

    plan: list[dict] = []
    total = 0

    for super_type in TEST_SUPER_ORDER:
        banks = TEST_BANKS.get(super_type, {})
        if not isinstance(banks, dict):
            continue

        for sub_type in sorted(list(banks.keys())):
            words = banks.get(sub_type, [])
            if not isinstance(words, list):
                continue

            picked = _pick_n_words([w for w in words if isinstance(w, str) and w.strip()], n)
            if not picked:
                continue

            plan.append({"super_type": super_type, "sub_type": sub_type, "words": picked})
            total += len(picked)

    if total == 0:
        raise HTTPException(status_code=500, detail="Test banks empty / failed to build plan")

    flat_queue = build_test_flat_queue(plan)
    plan_obj = {
        "language": TEST_LANG,
        "words_per_subtype": n,
        "plan": plan,
        "flatQueue": flat_queue,
        "total_questions": total,
    }

    config_obj = {
        "enabled_super_order": TEST_SUPER_ORDER,
        "enabled_subtypes": {st: sorted(list((TEST_BANKS.get(st) or {}).keys())) for st in TEST_SUPER_ORDER},
    }

    # ✅ 這裡先 insert tests（plan NOT NULL）
    test_id = create_test_session_row_v2(
        user_id=current_user_id,
        test_name=f"SLP Screening ({TEST_LANG})",
        language=TEST_LANG,
        words_per_subtype=n,
        threshold=TEST_SLP_THRESHOLD,
        plan_obj=plan_obj,
        total_questions=total,
        config_obj=config_obj,
    )

    return TestStartResp(
        test_id=test_id,
        language=TEST_LANG,
        words_per_subtype=n,
        plan=plan,
        total_questions=total,
    )

class TestFinishResp(BaseModel):
    test_id: str
    language: str
    threshold: int
    confirmed: List[Dict[str, Any]]  # [{super_type, sub_type, slp_count}]
    has_slp: int

class TestAnalyzeResp(BaseModel):
    test_record_id: str
    test_id: str
    super_type: str
    sub_type: str
    target_word: str

    asr_ipa: str
    revise_ipa: str
    asr_word: str
    correct_rate: int
    status: str
    severity: Optional[str] = None
    is_off_target: bool = False
    check_results: dict


def _detector_key_for_super(super_type: str) -> str:
    mapping = {
        "Addition": "addition",
        "Distortion": "distortion",
        "Omission": "omission",
        "Substitution": "substitution",
        "Tone": "tone",
        "VowelFinal": "vowel_final",
        "InitialConsonant": "initial",
    }
    return mapping.get(super_type, "")


def _run_detector(super_type: str, target_word: str, asr_ipa: str, off: dict, correct_rate: int) -> dict:
    """
    Run the matching rule-based detector (no LLM) and return its dict result.
    """
    st = (super_type or "").strip()

    if st == "Addition":
        return detect_addition_error_1(target_word, asr_ipa, llm=None, off_target=off, correct_rate=correct_rate)

    if st == "Distortion":
        return detect_distortion_error(target_word, asr_ipa, llm=None, off_target=off, correct_rate=correct_rate)

    if st == "Omission":
        return detect_omission_error_1(target_word, asr_ipa, llm=None, off_target=off, correct_rate=correct_rate)

    if st == "Substitution":
        return detect_substitution_error_1(target_word, asr_ipa, llm=None, off_target=off, correct_rate=correct_rate)

    if st == "Tone":
        return detect_tone_error(target_word, asr_ipa, llm=None, off_target=off, correct_rate=correct_rate)

    if st == "VowelFinal":
        return detect_vowelfinal_error(target_word, asr_ipa, llm=None, off_target=off, correct_rate=correct_rate)

    if st == "InitialConsonant":
        return detect_initial_consonant_error(target_word, asr_ipa, llm=None, off_target=off, correct_rate=correct_rate)

    raise HTTPException(status_code=400, detail=f"Unsupported super_type: {super_type}")


def _normalize_severity_value(v) -> Optional[str]:
    s = str(v or "").strip().lower()
    if s in ("slight", "middle", "high"):
        return s
    if s == "medium":
        return "middle"
    return None


@app.post("/api/test/analyze", response_model=TestAnalyzeResp)
async def api_test_analyze(
    file: UploadFile = File(...),
    test_id: str = Form(...),
    super_type: str = Form(...),
    sub_type: str = Form(...),
    target_word: str = Form(...),
    question_index: int = Form(...),
    difficulty_level: str = Form("小學"),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Test-only analyze:
    - runs ASR + single detector based on super_type
    - saves into test_records
    - does NOT touch records_analysis
    """
    super_type = (super_type or "").strip()
    sub_type = (sub_type or "").strip()
    target_word = (target_word or "").strip()

    if not test_id:
        raise HTTPException(status_code=400, detail="Missing test_id")
    if not super_type or super_type not in TEST_SUPER_ORDER:
        raise HTTPException(status_code=400, detail="Invalid super_type")
    if not sub_type:
        raise HTTPException(status_code=400, detail="Missing sub_type")
    if not target_word:
        raise HTTPException(status_code=400, detail="Missing target_word")

    # verify test_id belongs to this patient
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT patient_id FROM tests WHERE id=?", (test_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="test_id not found")
    if row[0] != current_user_id:
        raise HTTPException(status_code=403, detail="Not allowed")

    # verify sub_type exists in bank for this super_type (safety)
    bank_map = TEST_BANKS.get(super_type, {})
    if sub_type not in bank_map:
        raise HTTPException(status_code=400, detail=f"sub_type not in bank: {sub_type}")

    content = await file.read()

    # ASR
    asr_result = transcribe_bytes(
        audio_bytes=content,
        filename=file.filename,
        target_word=target_word,
        difficulty_level=difficulty_level,
    )

    asr_ipa = asr_result.get("asr_ipa", "") or ""
    target_ipa = (asr_result.get("target_ipa", "") or "").replace("/", "") or ""

    asr_word = asr_result.get("asr_word", "") or ""
    if not asr_word:
        if asr_ipa.strip() == target_ipa.strip():
            asr_word = target_word
        else:
            asr_word = pinyin_to_chinese_diff(asr_ipa, target_ipa, target_word)

    # accuracy + score
    correct_rate = calc_correct_rate(asr_ipa, target_ipa)
    score_points = int(round(correct_rate / 10))

    # off target
    off = compute_off_target(target_ipa, asr_ipa)
    is_off_target = bool(off.get("is_off_target"))

    # detector
    det_res = _run_detector(super_type, target_word, asr_ipa, off, correct_rate)
    det_key = _detector_key_for_super(super_type)
    if not det_key:
        raise HTTPException(status_code=500, detail="detector key mapping missing")

    # build check_results (include test meta)
    check_results = {
        "meta": {
            "event": "test_analyze",
            "test_id": test_id,
            "language": TEST_LANG,
            "off_target": off,
            "question_index": int(question_index),
            "enabled_checks": [det_key],
            "super_type": super_type,
            "sub_type": sub_type,
        },
        det_key: det_res,
    }

    # status = ipa match (same convention you already use)
    status = "correct" if (asr_ipa.strip() == target_ipa.strip()) else "wrong"

    # severity only when final_label == slp
    final_label = str((det_res or {}).get("final_label") or "").strip().lower()
    severity_value = _normalize_severity_value((det_res or {}).get("severity")) if final_label == "slp" else None

    # ✅ Save test_record + update progress_cursor in ONE transaction
    conn = get_db()
    cur = conn.cursor()
    try:
        # Ensure test still belongs to this user and is running
        cur.execute("SELECT patient_id, status FROM tests WHERE id=?", (test_id,))
        trow = cur.fetchone()
        if not trow:
            raise HTTPException(status_code=404, detail="test_id not found")
        if trow[0] != current_user_id:
            raise HTTPException(status_code=403, detail="Not allowed")
        if str(trow[1] or "").lower() != "running":
            raise HTTPException(status_code=400, detail="Test is not running")

        # 1) insert into test_records (same tx)
        test_record_id = save_test_record_v2_tx(
            cur,
            test_id=test_id,
            question_index=int(question_index),
            user_id=current_user_id,
            language="Chinese",
            test_type=super_type,
            category=sub_type,
            target_word=target_word,
            target_ipa=target_ipa,
            asr_ipa=asr_ipa,
            asr_word=asr_word,
            correct_rate=correct_rate,
            check_results=check_results,
            difficulty_level=difficulty_level,
            score=score_points,
            severity=severity_value,
        )

        # 2) recompute cursor (now includes this inserted row)
        cur.execute("SELECT COUNT(DISTINCT question_index) FROM test_records WHERE test_id=?", (test_id,))
        done_count = int(cur.fetchone()[0] or 0)

        cur.execute("""
            UPDATE tests
            SET progress_cursor=?,
                current_question_index=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (done_count, done_count, test_id))

        conn.commit()

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"test_analyze failed: {str(e)}")
    finally:
        conn.close()

    return TestAnalyzeResp(
        test_record_id=test_record_id,
        test_id=test_id,
        super_type=super_type,
        sub_type=sub_type,
        target_word=target_word,
        asr_ipa=asr_ipa,
        revise_ipa=target_ipa,
        asr_word=asr_word,
        correct_rate=correct_rate,
        status=status,
        severity=severity_value,
        is_off_target=is_off_target,
        check_results=check_results,
    )

@app.post("/api/test/finish", response_model=TestFinishResp)
def api_test_finish(
    test_id: str = Query(...),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Finish test:
    - read test_records for this test_id (from check_results.meta.test_id)
    - count slp per (super_type, sub_type)
    - if count >= threshold => insert into slp_patient
    - update tests.has_slp and tests.result
    """
    if not test_id:
        raise HTTPException(status_code=400, detail="Missing test_id")

    # verify ownership
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT patient_id FROM tests WHERE id=?", (test_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="test_id not found")
    if row[0] != current_user_id:
        conn.close()
        raise HTTPException(status_code=403, detail="Not allowed")

    # load records for this test_id
    cur.execute("""
        SELECT test_type, category, check_results
        FROM test_records
        WHERE test_id=?
        ORDER BY question_index ASC
    """, (test_id,))
    rows = cur.fetchall()
    conn.close()

    # count slp by (super_type, sub_type)
    counter = {}  # (st, sb) -> count
    for st, sb, cr_json in rows:
        try:
            obj = json.loads(cr_json) if cr_json else {}
        except Exception:
            continue

        meta = obj.get("meta", {}) if isinstance(obj.get("meta"), dict) else {}
        if meta.get("test_id") != test_id:
            continue

        super_type = str(meta.get("super_type") or st or "").strip()
        sub_type = str(meta.get("sub_type") or sb or "").strip()
        if not super_type or not sub_type:
            continue

        det_key = _detector_key_for_super(super_type)
        det = obj.get(det_key, {}) if det_key else {}
        final_label = str((det or {}).get("final_label") or "").strip().lower()

        if final_label == "slp":
            counter[(super_type, sub_type)] = counter.get((super_type, sub_type), 0) + 1

    confirmed = []
    confirmed_pairs = []
    for (st, sb), c in sorted(counter.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        if c >= TEST_SLP_THRESHOLD:
            confirmed.append({"super_type": st, "sub_type": sb, "slp_count": c})
            confirmed_pairs.append((st, sb))

    # insert slp_patient rows
    insert_slp_patient_rows(
        user_id=current_user_id,
        test_id=test_id,
        confirmed=confirmed_pairs,
        language=TEST_LANG
    )

    has_slp = 1 if confirmed_pairs else 0

    # update tests row
    summary_obj = {
        "language": TEST_LANG,
        "threshold": TEST_SLP_THRESHOLD,
        "confirmed": confirmed,
        "counts": {f"{k[0]}::{k[1]}": v for k, v in counter.items()},
    }
    update_test_session_result(test_id=test_id, has_slp=has_slp, result_obj=summary_obj)

    return TestFinishResp(
        test_id=test_id,
        language=TEST_LANG,
        threshold=TEST_SLP_THRESHOLD,
        confirmed=confirmed,
        has_slp=has_slp
    )

class TestResumeResp(BaseModel):
    available: bool
    test_id: Optional[str] = None
    status: Optional[str] = None
    progress_cursor: int = 0
    plan: Optional[dict] = None
    language: Optional[str] = None

@app.get("/api/test/resume", response_model=TestResumeResp)
def api_test_resume(
    language: str = Query("cn"),
    current_user_id: str = Depends(get_current_user_id)
):
    ensure_tests_table_v2_columns()

    lang = normalize_practice_language(language)  # 會變成 'cn' 或 'en'

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, status, progress_cursor, plan, language
            FROM tests
            WHERE patient_id=?
              AND status='running'
              AND language=?                     -- ✅ 關鍵：限制語言
            ORDER BY datetime(updated_at) DESC, datetime(date) DESC
            LIMIT 1
        """, (current_user_id, lang))
        row = cur.fetchone()

        if not row:
            return TestResumeResp(available=False)

        test_id, status, progress_cursor, plan_json, test_lang = row

        plan_obj = None
        if plan_json:
            try:
                plan_obj = json.loads(plan_json)
            except Exception:
                plan_obj = {"raw": plan_json}

        return TestResumeResp(
            available=True,
            test_id=test_id,
            status=status,
            progress_cursor=int(progress_cursor or 0),
            plan=plan_obj,
            language=test_lang
        )
    finally:
        conn.close()

def save_test_record_v2_tx(
    cur,
    *,
    test_id: str,
    question_index: int,
    user_id: str,
    language: str,
    test_type: str,
    category: str,
    target_word: str,
    target_ipa: str,
    asr_ipa: str,
    asr_word: str,
    correct_rate,
    check_results: dict,
    difficulty_level: str = "primary",
    score: int | None = None,
    severity: str | None = None,
    audio_path: str | None = None,
) -> str:
    rid = str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.execute("""
        INSERT INTO test_records (
            id,
            test_id, user_id, question_index,
            date,
            target_word, target_ipa, asr_ipa, asr_word,
            score, correct_rate, severity, difficulty,
            check_results, language, test_type, category,
            audio_path,
            created_at, updated_at
        ) VALUES (
            ?,
            ?, ?, ?,
            ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?,
            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
        )
    """, (
        rid,
        test_id, user_id, int(question_index),
        now,
        target_word, target_ipa, asr_ipa, asr_word,
        score,
        str(correct_rate) if correct_rate is not None else None,
        severity,
        normalize_difficulty(difficulty_level),
        json.dumps(check_results, ensure_ascii=False),
        language,
        test_type,
        category,
        audio_path,
    ))
    return rid

@app.get("/api/training/confirmed")
def api_training_confirmed(
    test_id: str = Query(None),
    current_user_id: str = Depends(get_current_user_id),
):
    conn = get_db()
    try:
        pairs = slp_tasks.get_confirmed_slp_pairs(conn, user_id=current_user_id, test_id=test_id)
        return {"items": [{"super_type": a, "sub_type": b} for a, b in pairs], "count": len(pairs)}
    finally:
        conn.close()

from pydantic import BaseModel
from typing import List, Optional

class ChooseTwoReq(BaseModel):
    test_id: Optional[str] = None
    chosen: List[dict]
    language: Optional[str] = None  # ✅ add

@app.post("/api/training/allocate/init")
def api_training_allocate_init(
    test_id: str = Query(...),
    language: Optional[str] = Query(None),   # ✅ add this
    current_user_id: str = Depends(get_current_user_id),
):
    conn = get_db()
    try:
        # ✅ normalize language (only cn/en)
        L = (language or "").strip().lower()
        if L in ("zh", "zh-cn", "zh_tw", "zh-tw", "chinese", "cn"):
            L = "cn"
        elif L in ("en", "english"):
            L = "en"
        else:
            L = "cn"  # default / fallback

        pairs = slp_tasks.get_confirmed_slp_pairs(conn, user_id=current_user_id, test_id=test_id, language=L)  # ✅ (if you added language filter)

        out = slp_tasks.allocate_training_tasks_for_confirmed(
            conn,
            user_id=current_user_id,
            confirmed_pairs=pairs,
            language=L,              # ✅ use selected language
            task_kind="vocab",
            chosen_pairs=[],         # init: not chosen yet
            preferred_package=(30, 3),
            decreased_package=(20, 2),
        )
        return {"msg": "ok", **out}
    finally:
        conn.close()

class TrainingCommitReq(BaseModel):
    task_id: str
    increment: int = 1

@app.post("/api/training/commit")
def api_training_commit(
    data: TrainingCommitReq,
    current_user_id: str = Depends(get_current_user_id),
):
    conn = get_db()
    try:
        return slp_tasks.commit_training_task_progress(
            conn,
            user_id=current_user_id,
            task_id=data.task_id,
            increment=data.increment,
            daily_cap=6,   # 之後改成可配置
        )
    finally:
        conn.close()

class TrainingBatchTask(BaseModel):
    id: str
    user_id: str
    super_type: Optional[str] = None
    sub_type: Optional[str] = None
    task_kind: str
    language: str
    allocated_at: Optional[str] = None
    day_index: int
    total_times: int
    progress_times: int
    status: str
    is_active: int

class TrainingBatchResp(BaseModel):
    available: bool
    super_type: str
    sub_type: str
    language: str
    task_kind: str
    allocated_at: Optional[str] = None
    allowed_day_index: Optional[int] = None
    tasks: List[TrainingBatchTask] = []
    reason: Optional[str] = None


@app.get("/api/tasks/training/batch", response_model=TrainingBatchResp)
def api_tasks_training_batch(
    super_type: str = Query(...),
    sub_type: str = Query(...),
    language: str = Query("cn"),
    task_kind: str = Query("vocab"),  # vocab | sentence
    current_user_id: str = Depends(get_current_user_id),
):
    lang = normalize_practice_language(language)
    super_type = (super_type or "").strip()
    sub_type = (sub_type or "").strip()
    task_kind = (task_kind or "vocab").strip().lower()

    if task_kind not in ("vocab", "sentence"):
        raise HTTPException(status_code=400, detail="Invalid task_kind (vocab|sentence)")

    conn = get_db()
    cur = conn.cursor()
    try:
        # 1) 找最新一批 allocated_at
        cur.execute("""
            SELECT allocated_at
            FROM tasks
            WHERE user_id=?
              AND super_type=?
              AND sub_type=?
              AND language=?
              AND task_kind=?
              AND is_active=1
              AND allocated_at IS NOT NULL
            ORDER BY datetime(allocated_at) DESC
            LIMIT 1
        """, (current_user_id, super_type, sub_type, lang, task_kind))
        r = cur.fetchone()
        if not r or not r[0]:
            return TrainingBatchResp(
                available=False,
                super_type=super_type,
                sub_type=sub_type,
                language=lang,
                task_kind=task_kind,
                allocated_at=None,
                allowed_day_index=None,
                tasks=[],
                reason="NO_BATCH"
            )

        allocated_at = r[0]
        allowed = _task_allowed_day_index(allocated_at)

        # 2) 取這批 day1..day7 tasks
        cur.execute("""
            SELECT
              id, user_id, super_type, sub_type, task_kind, language,
              allocated_at, day_index, total_times, progress_times, status, is_active
            FROM tasks
            WHERE user_id=?
              AND super_type=?
              AND sub_type=?
              AND language=?
              AND task_kind=?
              AND is_active=1
              AND datetime(allocated_at)=datetime(?)
            ORDER BY day_index ASC
        """, (current_user_id, super_type, sub_type, lang, task_kind, allocated_at))
        rows = cur.fetchall()

        tasks = []
        for row in rows:
            (tid, uid, st, sb, kind, lg, alloc, day_idx,
             total, prog, status, is_active) = row

            tasks.append(TrainingBatchTask(
                id=tid,
                user_id=uid,
                super_type=st,
                sub_type=sb,
                task_kind=kind,
                language=lg,
                allocated_at=alloc,
                day_index=int(day_idx or 1),
                total_times=int(total or 0),
                progress_times=int(prog or 0),
                status=str(status or "pending"),
                is_active=int(is_active or 0),
            ))

        return TrainingBatchResp(
            available=True,
            super_type=super_type,
            sub_type=sub_type,
            language=lang,
            task_kind=task_kind,
            allocated_at=str(allocated_at),
            allowed_day_index=allowed,
            tasks=tasks,
            reason=None
        )
    finally:
        conn.close()

@app.post("/api/training/allocate/choose-two")
def api_training_allocate_choose_two(
    data: ChooseTwoReq,
    current_user_id: str = Depends(get_current_user_id),
):
    chosen_pairs = []
    for it in (data.chosen or []):
        st = str((it or {}).get("super_type") or "").strip()
        sb = str((it or {}).get("sub_type") or "").strip()
        if st and sb:
            chosen_pairs.append((st, sb))

    # 去重並限制最多 2
    uniq = []
    seen = set()
    for p in chosen_pairs:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    chosen_pairs = uniq[:2]

    # ✅ normalize language (cn/en)
    L = (data.language or "").strip().lower()
    if L in ("zh", "zh-cn", "zh_tw", "zh-tw", "chinese", "cn"):
        L = "cn"
    elif L in ("en", "english"):
        L = "en"
    else:
        L = "cn"  # default

    conn = get_db()
    try:
        # ✅ 建議：confirmed_pairs 亦用同一語言（避免撈到 CN/EN 混埋）
        confirmed_pairs = slp_tasks.get_confirmed_slp_pairs(
            conn, user_id=current_user_id, test_id=data.test_id, language=L
        )

        confirmed_set = set(confirmed_pairs)
        for p in chosen_pairs:
            if p not in confirmed_set:
                raise HTTPException(status_code=400, detail=f"Chosen pair not confirmed: {p[0]}/{p[1]}")

        out = slp_tasks.allocate_training_tasks_for_confirmed(
            conn,
            user_id=current_user_id,
            confirmed_pairs=confirmed_pairs,
            language=L,  # ✅ FIX: 用 L，唔好寫死 "cn"
            task_kind="vocab",
            chosen_pairs=chosen_pairs,
            preferred_package=(30, 3),
            decreased_package=(20, 2),
        )
        return {"msg": "ok", "chosen": [{"super_type": a, "sub_type": b} for a, b in chosen_pairs], **out}
    finally:
        conn.close()

class TrainingTaskRow(BaseModel):
    id: str
    super_type: Optional[str] = None
    sub_type: Optional[str] = None
    task_kind: str
    language: str
    allocated_at: Optional[str] = None
    day_index: int
    total_times: int
    progress_times: int
    status: str
    is_active: int
    reward_claimed: int = 0
    reward_claimed_at: Optional[str] = None

class TrainingAllActiveResp(BaseModel):
    available: bool
    latest_allocated_at: Optional[str] = None
    latest_allowed_day_index: Optional[int] = None
    tasks: List[TrainingTaskRow] = []
    reason: Optional[str] = None

class ClaimRewardReq(BaseModel):
    task_id: str

class ClaimRewardResp(BaseModel):
    msg: str
    task_id: str
    added: int
    integral: int

@app.post("/api/tasks/training/claim-reward", response_model=ClaimRewardResp)
def api_tasks_training_claim_reward(
    data: ClaimRewardReq,
    current_user_id: str = Depends(get_current_user_id),
):
    ensure_task_table_v2()  # ✅ make sure reward columns exist

    task_id = (data.task_id or "").strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="Missing task_id")

    conn = get_db()
    cur = conn.cursor()
    try:
        # 1) verify task exists + ownership
        cur.execute("""
            SELECT id, user_id, reward_claimed, status, total_times, progress_times
            FROM tasks
            WHERE id=?
            LIMIT 1
        """, (task_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")

        tid, uid, claimed, status, total_times, progress_times = row
        if uid != current_user_id:
            raise HTTPException(status_code=403, detail="Not allowed")

        if int(claimed or 0) == 1:
            raise HTTPException(status_code=400, detail="Reward already claimed")

        # ✅ 必須完成任務才可領取
        st = str(status or "").lower()
        total = int(total_times or 0)
        prog = int(progress_times or 0)

        # 你的 task 有時 total_times 可能是 0（不正常），這裡也擋掉
        if total <= 0:
            raise HTTPException(status_code=400, detail="Task has invalid total_times")

        # 必須 completed + 進度滿
        if st != "completed" or prog < total:
            raise HTTPException(status_code=400, detail="Task not completed yet")
        # 2) atomic claim (prevents double click / race condition)
        cur.execute("""
            UPDATE tasks
            SET reward_claimed=1,
                reward_claimed_at=CURRENT_TIMESTAMP
            WHERE id=?
            AND user_id=?
            AND reward_claimed=0
            AND LOWER(status)='completed'
            AND COALESCE(progress_times,0) >= COALESCE(total_times,0)
            AND COALESCE(total_times,0) > 0
        """, (tid, current_user_id))
        if cur.rowcount != 1:
            raise HTTPException(status_code=400, detail="Reward not claimable (already claimed or not completed)")

        # 3) add points
        added = 50
        cur.execute("""
            UPDATE patients
            SET integral = COALESCE(integral,0) + ?
            WHERE id=?
        """, (added, current_user_id))

        # 4) return latest integral
        cur.execute("SELECT COALESCE(integral,0) FROM patients WHERE id=?", (current_user_id,))
        integral = int(cur.fetchone()[0] or 0)

        conn.commit()
        return {"msg": "ok", "task_id": tid, "added": added, "integral": integral}

    finally:
        conn.close()

@app.get("/api/tasks/training/all-active", response_model=TrainingAllActiveResp)
def api_tasks_training_all_active(
    current_user_id: str = Depends(get_current_user_id),
):
    conn = get_db()
    cur = conn.cursor()
    try:
        # 1) 找最新 allocated_at（用來算今天 day）
        cur.execute("""
            SELECT allocated_at
            FROM tasks
            WHERE user_id=?
              AND is_active=1
              AND allocated_at IS NOT NULL
              AND super_type IS NOT NULL
              AND sub_type IS NOT NULL
            ORDER BY datetime(allocated_at) DESC
            LIMIT 1
        """, (current_user_id,))
        r = cur.fetchone()
        latest_alloc = r[0] if r else None
        latest_allowed = _task_allowed_day_index(latest_alloc) if latest_alloc else None

        # ✅ FIX: Day8+ (expired) => treat as NO active training for today; ask for new SLP test
        # _task_allowed_day_index returns None when days_since <0 or >6
        if latest_alloc and latest_allowed is None:
            return TrainingAllActiveResp(
                available=False,
                latest_allocated_at=latest_alloc,
                latest_allowed_day_index=None,
                tasks=[],
                reason="NEED_NEW_SLP_TEST"
            )

        # 2) 取全部 active training tasks
        cur.execute("""
            SELECT
              id, super_type, sub_type, task_kind, language,
              allocated_at, day_index, total_times, progress_times, status, is_active,
              reward_claimed, reward_claimed_at
            FROM tasks
            WHERE user_id=?
              AND is_active=1
              AND super_type IS NOT NULL
              AND sub_type IS NOT NULL
            ORDER BY
              datetime(allocated_at) DESC,
              super_type ASC,
              sub_type ASC,
              language ASC,
              task_kind ASC,
              day_index ASC
        """, (current_user_id,))
        rows = cur.fetchall()

        tasks = []
        for row in rows:
            (tid, st, sb, kind, lang, alloc, day_idx, total, prog, status, is_active, reward_claimed, reward_claimed_at) = row
            tasks.append(TrainingTaskRow(
                id=tid,
                super_type=st,
                sub_type=sb,
                task_kind=str(kind or "vocab"),
                language=str(lang or "cn"),
                allocated_at=alloc,
                day_index=int(day_idx or 1),
                total_times=int(total or 0),
                progress_times=int(prog or 0),
                status=str(status or "pending"),
                is_active=int(is_active or 0),
                reward_claimed=int(reward_claimed or 0),
                reward_claimed_at=reward_claimed_at,
            ))

        # 3) ✅ 沒有任何 active training tasks
        if not tasks:
            # ✅ 是否曾經有 CN training（代表「做完一輪」或「曾經 allocate 過」）
            cur.execute("""
                SELECT 1
                FROM tasks
                WHERE user_id=?
                  AND language='cn'
                  AND task_kind='vocab'
                  AND allocated_at IS NOT NULL
                  AND super_type IS NOT NULL
                  AND sub_type IS NOT NULL
                LIMIT 1
            """, (current_user_id,))
            had_any_cn_training = cur.fetchone() is not None

            return TrainingAllActiveResp(
                available=False,
                latest_allocated_at=latest_alloc,
                latest_allowed_day_index=latest_allowed,
                tasks=[],
                reason="NEED_NEW_SLP_TEST" if had_any_cn_training else "NO_ACTIVE_TRAINING_TASKS"
            )

        # 4) 正常回傳（Day1-7）
        return TrainingAllActiveResp(
            available=True,
            latest_allocated_at=latest_alloc,
            latest_allowed_day_index=latest_allowed,
            tasks=tasks,
            reason=None
        )
    finally:
        conn.close()

def deactivate_batch_if_fully_completed(cur, *, user_id: str, super_type: str, sub_type: str, language: str, task_kind: str, allocated_at: str):
    """
    如果同一批 allocated_at 的 day1..day7 全部 completed，將該批 tasks.is_active=0
    """
    cur.execute("""
        SELECT COUNT(*) AS total_cnt,
               SUM(CASE WHEN LOWER(status)='completed' THEN 1 ELSE 0 END) AS done_cnt
        FROM tasks
        WHERE user_id=?
          AND super_type=?
          AND sub_type=?
          AND language=?
          AND task_kind=?
          AND is_active=1
          AND allocated_at IS NOT NULL
          AND datetime(allocated_at)=datetime(?)
          AND day_index BETWEEN 1 AND 7
    """, (user_id, super_type, sub_type, language, task_kind, allocated_at))
    row = cur.fetchone()
    total_cnt = int(row[0] or 0)
    done_cnt = int(row[1] or 0)

    # 必須有 7 筆，而且全 completed
    if total_cnt >= 7 and done_cnt >= 7:
        cur.execute("""
            UPDATE tasks
            SET is_active=0
            WHERE user_id=?
              AND super_type=?
              AND sub_type=?
              AND language=?
              AND task_kind=?
              AND is_active=1
              AND datetime(allocated_at)=datetime(?)
              AND day_index BETWEEN 1 AND 7
        """, (user_id, super_type, sub_type, language, task_kind, allocated_at))

from pydantic import BaseModel

class TestReportItem(BaseModel):
    id: str
    test_name: Optional[str] = None
    status: str
    language: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    progress_cursor: int = 0

class TestReportsResp(BaseModel):
    items: List[TestReportItem]

@app.get("/api/test/reports", response_model=TestReportsResp)
def api_test_reports(
    limit: int = Query(80),
    language: Optional[str] = Query(None, description="cn|en (optional)"),
    current_user_id: str = Depends(get_current_user_id),
):
    conn = get_db()
    cur = conn.cursor()
    try:
        where_lang = ""
        params = [current_user_id]

        # normalize to your practice language convention
        lang = None
        if language is not None and str(language).strip() != "":
            lang = normalize_practice_language(language)  # returns "cn" or "en"
            where_lang = " AND language=?"
            params.append(lang)

        params.append(int(limit))

        cur.execute(f"""
            SELECT
                id,
                test_name,
                status,
                language,
                date,
                updated_at,
                progress_cursor
            FROM tests
            WHERE patient_id=?
            {where_lang}
            ORDER BY datetime(updated_at) DESC, datetime(date) DESC
            LIMIT ?
        """, tuple(params))

        rows = cur.fetchall()
        items = []
        for r in rows:
            items.append({
                "id": r[0],
                "test_name": r[1],
                "status": r[2] or "unknown",
                "language": r[3],
                "created_at": r[4],
                "updated_at": r[5],
                "progress_cursor": int(r[6] or 0),
            })

        return {"items": items}
    finally:
        conn.close()

@app.get("/api/training/confirmed-pairs")
def api_training_confirmed_pairs_alias(
    test_id: Optional[str] = Query(None),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Alias for frontend compatibility.
    Returns confirmed (super_type, sub_type) pairs for a given test_id (optional).
    """
    conn = get_db()
    try:
        pairs = slp_tasks.get_confirmed_slp_pairs(conn, user_id=current_user_id, test_id=test_id)
        return {"items": [{"super_type": a, "sub_type": b} for a, b in pairs]}
    finally:
        conn.close()
        
@app.get("/api/patient/tests/stats")
def api_patient_tests_stats(
    days: int = Query(365, ge=1, le=3650, description="default 365"),
    date_from: Optional[str] = Query(None, alias="from", description="YYYY-MM-DD (optional)"),
    date_to: Optional[str] = Query(None, alias="to", description="YYYY-MM-DD (optional)"),
    language: Optional[str] = Query(None, description="cn|en (optional)"),
    a: Optional[str] = Query(None, description="test_id A for compare (older)"),
    b: Optional[str] = Query(None, description="test_id B for compare (newer)"),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Return aggregated finished test stats for the current patient.
    - includes list of tests with aggregated metrics from test_records + slp_patient
    - includes default compare: latest_finished vs prev_finished (if >=2)
    - optional compare for any a/b test_id
    """
    # range
    d1_def, d2_def = _ymd_default_range(days)
    d1 = _parse_ymd_or_default(date_from, d1_def)
    d2 = _parse_ymd_or_default(date_to, d2_def)

    # lang
    where_lang = ""
    params: list[Any] = [current_user_id, d1, d2]
    if language is not None and str(language).strip() != "":
        lang = normalize_practice_language(language)  # your helper returns cn|en
        where_lang = " AND language=?"
        params.append(lang)

    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        # 1) fetch finished tests in range
        cur.execute(f"""
            SELECT
              id, test_name, status, language, threshold, words_per_subtype,
              progress_cursor, total_questions, current_question_index,
              has_slp, date, updated_at, finished_at, result
            FROM tests
            WHERE patient_id=?
              AND status='finished'
              AND date(COALESCE(finished_at, updated_at, date)) BETWEEN date(?) AND date(?)
              {where_lang}
            ORDER BY datetime(COALESCE(finished_at, updated_at, date)) DESC
        """, tuple(params))
        tests = cur.fetchall()

        items: List[dict] = []

        for t in tests:
            test_id = t["id"]

            # 2) aggregate test_records
            cur.execute("""
                SELECT score, correct_rate, severity, check_results
                FROM test_records
                WHERE test_id=?
            """, (test_id,))
            rec_rows = cur.fetchall()
            recordsAgg = _test_agg_from_rows(rec_rows)

            # 3) aggregate slp_patient
            cur.execute("""
                SELECT super_type, sub_type
                FROM slp_patient
                WHERE test_id=?
            """, (test_id,))
            slp_rows = cur.fetchall()
            slpAgg = _slp_agg_from_rows(slp_rows)

            items.append({
                "test": {
                    "id": test_id,
                    "test_name": t["test_name"],
                    "status": t["status"],
                    "language": t["language"],
                    "threshold": int(t["threshold"] or 0),
                    "words_per_subtype": int(t["words_per_subtype"] or 0),
                    "progress_cursor": int(t["progress_cursor"] or 0),
                    "total_questions": int(t["total_questions"] or 0),
                    "current_question_index": int(t["current_question_index"] or 0),
                    "has_slp": int(t["has_slp"] or 0),
                    "created_at": t["date"],
                    "updated_at": t["updated_at"],
                    "finished_at": t["finished_at"],
                    "result": _safe_json_loads(t["result"]),
                },
                "recordsAgg": recordsAgg,
                "slpAgg": slpAgg,
            })

        # 4) default latest vs prev
        latest = items[0] if len(items) >= 1 else None
        prev = items[1] if len(items) >= 2 else None

        default_compare = None
        if latest and prev:
            default_compare = {
                "a": prev["test"]["id"],
                "b": latest["test"]["id"],
                "delta": _compare_metrics(prev, latest),
            }

        # 5) optional a/b compare
        custom_compare = None
        if a and b:
            # find in items first
            map_by_id = {it["test"]["id"]: it for it in items}
            if a not in map_by_id or b not in map_by_id:
                # fallback: verify ownership then load on demand
                for tid in (a, b):
                    cur.execute("SELECT patient_id FROM tests WHERE id=?", (tid,))
                    rr = cur.fetchone()
                    if not rr:
                        raise HTTPException(status_code=404, detail=f"test not found: {tid}")
                    if rr["patient_id"] != current_user_id:
                        raise HTTPException(status_code=403, detail="not allowed")

                def load_one(tid: str) -> dict:
                    cur.execute("""
                        SELECT id, test_name, status, language, threshold, words_per_subtype,
                               progress_cursor, total_questions, current_question_index,
                               has_slp, date, updated_at, finished_at, result
                        FROM tests WHERE id=? LIMIT 1
                    """, (tid,))
                    tt = cur.fetchone()
                    if not tt:
                        raise HTTPException(status_code=404, detail=f"test not found: {tid}")
                    cur.execute("SELECT score, correct_rate, severity, check_results FROM test_records WHERE test_id=?", (tid,))
                    rec_rows = cur.fetchall()
                    cur.execute("SELECT super_type, sub_type FROM slp_patient WHERE test_id=?", (tid,))
                    slp_rows = cur.fetchall()
                    return {
                        "test": {
                            "id": tt["id"],
                            "test_name": tt["test_name"],
                            "status": tt["status"],
                            "language": tt["language"],
                            "threshold": int(tt["threshold"] or 0),
                            "words_per_subtype": int(tt["words_per_subtype"] or 0),
                            "progress_cursor": int(tt["progress_cursor"] or 0),
                            "total_questions": int(tt["total_questions"] or 0),
                            "current_question_index": int(tt["current_question_index"] or 0),
                            "has_slp": int(tt["has_slp"] or 0),
                            "created_at": tt["date"],
                            "updated_at": tt["updated_at"],
                            "finished_at": tt["finished_at"],
                            "result": _safe_json_loads(tt["result"]),
                        },
                        "recordsAgg": _test_agg_from_rows(rec_rows),
                        "slpAgg": _slp_agg_from_rows(slp_rows),
                    }

                A = load_one(a)
                B = load_one(b)
            else:
                A = map_by_id[a]
                B = map_by_id[b]

            custom_compare = {
                "a": A["test"]["id"],
                "b": B["test"]["id"],
                "delta": _compare_metrics(A, B),
            }

        return {
            "range": {"from": d1, "to": d2},
            "items": items,
            "latest_finished": latest,
            "prev_finished": prev,
            "default_compare": default_compare,
            "custom_compare": custom_compare,
        }

    finally:
        conn.close()

@app.get("/api/patient/tests")
def api_patient_tests(
    limit: int = Query(80, ge=1, le=500),
    language: str | None = Query(None, description="cn|en (optional)"),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    List tests (super test table) for current patient.
    Exclude cancelled.
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        where_lang = ""
        params: list = [current_user_id]

        if language is not None and str(language).strip() != "":
            lang = normalize_practice_language(language)  # returns "cn" or "en"
            where_lang = " AND language=?"
            params.append(lang)

        params.append(int(limit))

        cur.execute(f"""
            SELECT
                id,
                test_name,
                status,
                language,
                threshold,
                words_per_subtype,
                progress_cursor,
                total_questions,
                current_question_index,
                has_slp,
                date,
                updated_at,
                finished_at
            FROM tests
            WHERE patient_id=?
              AND status <> 'cancelled'
              {where_lang}
            ORDER BY datetime(updated_at) DESC, datetime(date) DESC
            LIMIT ?
        """, tuple(params))

        rows = cur.fetchall()

        items = []
        for r in rows:
            items.append({
                "id": r[0],
                "test_name": r[1],
                "status": r[2],
                "language": r[3],
                "threshold": int(r[4] or 0),
                "words_per_subtype": int(r[5] or 0),
                "progress_cursor": int(r[6] or 0),
                "total_questions": int(r[7] or 0),
                "current_question_index": int(r[8] or 0),
                "has_slp": int(r[9] or 0),
                "created_at": r[10],
                "updated_at": r[11],
                "finished_at": r[12],
            })

        return {"items": items}
    finally:
        conn.close()

@app.get("/api/patient/tests/{test_id}")
def api_patient_test_detail(
    test_id: str,
    current_user_id: str = Depends(get_current_user_id),
):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                id, patient_id, test_name, status,
                progress_cursor, total_questions, current_question_index,
                plan, config_json,
                threshold, words_per_subtype, language,
                has_slp, result,
                date, updated_at, finished_at
            FROM tests
            WHERE id=?
              AND status <> 'cancelled'
            LIMIT 1
        """, (test_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="test not found")

        if row[1] != current_user_id:
            raise HTTPException(status_code=403, detail="not allowed")

        def try_json(s):
            if not s:
                return None
            try:
                return json.loads(s)
            except Exception:
                return {"raw": s}

        return {
            "test": {
                "id": row[0],
                "test_name": row[2],
                "status": row[3],
                "progress_cursor": int(row[4] or 0),
                "total_questions": int(row[5] or 0),
                "current_question_index": int(row[6] or 0),
                "plan": try_json(row[7]),
                "config": try_json(row[8]),
                "threshold": int(row[9] or 0),
                "words_per_subtype": int(row[10] or 0),
                "language": row[11],
                "has_slp": int(row[12] or 0),
                "result": try_json(row[13]),
                "created_at": row[14],
                "updated_at": row[15],
                "finished_at": row[16],
            }
        }
    finally:
        conn.close()

@app.get("/api/patient/tests/{test_id}/records")
def api_patient_test_records(
    test_id: str,
    current_user_id: str = Depends(get_current_user_id),
):
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # verify ownership + not cancelled
        cur.execute("SELECT patient_id, status FROM tests WHERE id=? LIMIT 1", (test_id,))
        t = cur.fetchone()
        if not t:
            raise HTTPException(status_code=404, detail="test not found")
        if str(t["status"] or "").lower() == "cancelled":
            raise HTTPException(status_code=404, detail="test cancelled")
        if t["patient_id"] != current_user_id:
            raise HTTPException(status_code=403, detail="not allowed")

        cur.execute("""
            SELECT *
            FROM test_records
            WHERE test_id=?
            ORDER BY question_index ASC, datetime(created_at) ASC
        """, (test_id,))
        rows = cur.fetchall()

        return {"items": [dict(r) for r in rows]}
    finally:
        conn.close()

from fastapi import Query, Depends, HTTPException

@app.get("/api/patient/tests/{test_id}/slp")
def api_patient_test_slp(
    test_id: str,
    language: str | None = Query(None, description="cn|en (optional)"),
    current_user_id: str = Depends(get_current_user_id),
):
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # verify ownership + not cancelled
        cur.execute("SELECT patient_id, status FROM tests WHERE id=? LIMIT 1", (test_id,))
        t = cur.fetchone()
        if not t:
            raise HTTPException(status_code=404, detail="test not found")
        if str(t["status"] or "").lower() == "cancelled":
            raise HTTPException(status_code=404, detail="test cancelled")
        if t["patient_id"] != current_user_id:
            raise HTTPException(status_code=403, detail="not allowed")

        where_lang = ""
        params = [test_id]

        if language is not None and str(language).strip() != "":
            lang = normalize_practice_language(language)  # "cn" or "en"
            where_lang = " AND language=?"
            params.append(lang)

        cur.execute(f"""
            SELECT
                id, user_id, test_id, datetime,
                super_type, sub_type, language, created_at
            FROM slp_patient
            WHERE test_id=?
            {where_lang}
            ORDER BY datetime(created_at) DESC, datetime(datetime) DESC
        """, tuple(params))

        rows = cur.fetchall()
        return {"items": [dict(r) for r in rows]}
    finally:
        conn.close()

from typing import Optional, Any, Dict, List, Tuple
from fastapi import Query, Depends, HTTPException
from datetime import datetime, timedelta
import json
import sqlite3

# ---------- helpers (stats) ----------

def _ymd_default_range(days: int = 365) -> tuple[str, str]:
    today = datetime.now().date()
    d2 = today.strftime("%Y-%m-%d")
    d1 = (today - timedelta(days=max(0, int(days) - 1))).strftime("%Y-%m-%d")
    return d1, d2

def _parse_ymd_or_default(s: Optional[str], fallback: str) -> str:
    if s is None or str(s).strip() == "":
        return fallback
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format, use YYYY-MM-DD")

def _safe_json_loads(s: Any) -> dict:
    if not s:
        return {}
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _detect_off_target_from_check_results(check_results_json: Any) -> bool:
    obj = _safe_json_loads(check_results_json)
    off = obj.get("meta", {}).get("off_target", {})
    if isinstance(off, dict) and off.get("is_off_target"):
        return True
    return False

def _normalize_sev_label(sev: Any) -> str:
    s = str(sev or "").strip().lower()
    if s in ("high", "severe", "hard"):
        return "high"
    if s in ("middle", "medium", "moderate"):
        return "middle"
    if s in ("slight", "low", "minor"):
        return "slight"
    if s in ("off_target", "offtarget", "off-target"):
        return "off_target"
    if not s:
        return "unknown"
    return "unknown"

def _safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(str(x)))
    except Exception:
        return None

def _avg(nums: List[float]) -> Optional[float]:
    if not nums:
        return None
    return sum(nums) / len(nums)

def _test_agg_from_rows(rows: List[sqlite3.Row]) -> dict:
    """
    rows: test_records rows for one test_id
    returns:
      {
        count,
        avg_correct_rate,
        avg_score,
        off_target_count,
        sev: {high,middle,slight,off_target,unknown},
      }
    """
    count = 0
    cr_list: List[float] = []
    score_list: List[float] = []
    off_target_count = 0
    sev = {"high": 0, "middle": 0, "slight": 0, "off_target": 0, "unknown": 0}

    for r in rows:
        count += 1

        # correct_rate stored as TEXT in db
        cr = _safe_int(r.get("correct_rate") if isinstance(r, dict) else r["correct_rate"])
        if cr is not None:
            cr_list.append(float(max(0, min(100, cr))))

        sc = _safe_int(r.get("score") if isinstance(r, dict) else r["score"])
        if sc is not None:
            score_list.append(float(sc))

        # off-target: prefer meta.off_target
        crj = r.get("check_results") if isinstance(r, dict) else r["check_results"]
        if _detect_off_target_from_check_results(crj):
            off_target_count += 1

        s = r.get("severity") if isinstance(r, dict) else r["severity"]
        lab = _normalize_sev_label(s)
        if lab not in sev:
            lab = "unknown"
        sev[lab] += 1

    avg_cr = _avg(cr_list)
    avg_sc = _avg(score_list)

    return {
        "count": count,
        "avg_correct_rate": None if avg_cr is None else round(avg_cr, 1),
        "avg_score": None if avg_sc is None else round(avg_sc, 2),
        "off_target_count": off_target_count,
        "off_target_rate": 0.0 if count == 0 else round(off_target_count * 100.0 / count, 1),
        "sev": sev,
    }

def _slp_agg_from_rows(rows: List[sqlite3.Row]) -> dict:
    """
    rows: slp_patient rows for one test_id
    returns:
      { confirmed_count, pairs: [{super_type, sub_type, count}] }
    """
    from collections import Counter
    c = Counter()
    for r in rows:
        st = (r.get("super_type") if isinstance(r, dict) else r["super_type"]) or ""
        sb = (r.get("sub_type") if isinstance(r, dict) else r["sub_type"]) or ""
        st = str(st).strip()
        sb = str(sb).strip()
        if st and sb:
            c[(st, sb)] += 1

    pairs = [
        {"super_type": k[0], "sub_type": k[1], "count": int(v)}
        for k, v in c.most_common()
    ]
    return {"confirmed_count": sum(c.values()), "pairs": pairs}

def _compare_metrics(a: dict, b: dict) -> dict:
    """
    a,b are items returned by stats list, containing recordsAgg & slpAgg.
    return delta summary for UI.
    """
    aR = a.get("recordsAgg") or {}
    bR = b.get("recordsAgg") or {}
    aS = a.get("slpAgg") or {}
    bS = b.get("slpAgg") or {}

    def fnum(x):
        try:
            return float(x)
        except Exception:
            return None

    def d(key, invert=False):
        av = fnum(aR.get(key))
        bv = fnum(bR.get(key))
        if av is None or bv is None:
            return None
        diff = bv - av
        return -diff if invert else diff

    delta = {
        "avg_correct_rate": d("avg_correct_rate", invert=False),   # higher better
        "avg_score": d("avg_score", invert=False),                 # higher better
        "off_target_rate": d("off_target_rate", invert=True),      # lower better => invert
        "confirmed_count": None,
        "sev_high": None,
        "sev_middle": None,
        "sev_slight": None,
    }

    # confirmed_count (lower better => invert)
    try:
        av = float(aS.get("confirmed_count") or 0)
        bv = float(bS.get("confirmed_count") or 0)
        delta["confirmed_count"] = -(bv - av)
    except Exception:
        delta["confirmed_count"] = None

    # severity counts (lower better => invert)
    try:
        aSev = aR.get("sev") or {}
        bSev = bR.get("sev") or {}
        for k in ("high", "middle", "slight"):
            av = float(aSev.get(k) or 0)
            bv = float(bSev.get(k) or 0)
            delta[f"sev_{k}"] = -(bv - av)
    except Exception:
        pass

    return delta

def _normalize_super_type_en(st: str) -> str:
    s = (st or "").strip()
    if s.lower() == "addition":
        return "Addition"
    if s.lower() == "omission":
        return "Omission"
    if s.lower() == "substitution":
        return "Substitution"
    return s

def _detector_key_for_super_en(super_type: str) -> str:
    # must match analyze_record_payload check_results keys
    mapping = {
        "Addition": "addition",
        "Omission": "omission",
        "Substitution": "substitution",
    }
    return mapping.get(super_type, "")

def _final_label_from_payload(super_type: str, payload: dict) -> str:
    """
    payload: output of analyze_record_payload
    expect payload.check_results.<detector_key>.final_label
    """
    det_key = _detector_key_for_super_en(super_type)
    if not det_key:
        return ""
    cr = (payload.get("check_results") or {})
    det = cr.get(det_key) if isinstance(cr, dict) else None
    if not isinstance(det, dict):
        return ""
    return str(det.get("final_label") or "").strip().lower()

def _pick_n_from_items(items: list[dict], n: int) -> list[dict]:
    """
    items: [{"w":"...", "ipa":"..."} ...]
    return random sample size n (or all if <n)
    """
    clean = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        w = (it.get("w") or "").strip()
        if not w:
            continue
        clean.append({"w": w, "ipa": (it.get("ipa") or "").strip()})
    if not clean:
        return []
    if len(clean) <= n:
        tmp = list(clean)
        random.shuffle(tmp)
        return tmp
    return random.sample(clean, n)

def build_test_flat_queue_en(plan: list[dict]) -> list[dict]:
    """
    plan: [{super_type, sub_type, words:[{w, ipa}]}]
    flatQueue: [{index, super_type, sub_type, word, ipa}]
    """
    flat = []
    idx = 0
    for blk in plan or []:
        st = str(blk.get("super_type") or "").strip()
        sb = str(blk.get("sub_type") or "").strip()
        words = blk.get("words") or []
        for it in words:
            if isinstance(it, dict):
                w = (it.get("w") or "").strip()
                ipa0 = (it.get("ipa") or "").strip()
            else:
                w = str(it).strip()
                ipa0 = ""
            if not w:
                continue
            flat.append({"index": idx, "super_type": st, "sub_type": sb, "word": w, "ipa": ipa0})
            idx += 1
    return flat

class TestStartENResp(BaseModel):
    test_id: str
    language: str
    words_per_subtype: int
    threshold: int
    plan: List[Dict[str, Any]]
    total_questions: int

@app.post("/api/test_en/start", response_model=TestStartENResp)
def api_test_en_start_real(
    words_per_subtype: int = Query(TEST_EN_WORDS_PER_SUBTYPE_DEFAULT, ge=1, le=20),
    threshold: int = Query(TEST_EN_SLP_THRESHOLD_DEFAULT, ge=1, le=10),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Create an EN test session:
    - read EN banks from LLM/*.json (NEW format)
    - for each super_type and each sub_type, pick N words (with IPA if provided)
    - save to tests.plan (NOT NULL) and return plan
    """
    n = int(words_per_subtype)
    th = int(threshold)

    # difficulty bucket for EN bank (primary|secondary|advanced)
    diff = infer_en_difficulty_from_user_id(current_user_id)

    plan: list[dict] = []
    total = 0

    enabled_subtypes: dict[str, list[str]] = {}

    for super_type in TEST_EN_SUPER_ORDER:
        super_type = _normalize_super_type_en(super_type)
        # read subtypes from bank file keys
        # use your existing helper: api_practice_en_subtypes logic
        path = EN_BANK_FILES.get(super_type)
        if not path:
            continue

        obj = safe_load_json(path)
        banks = obj.get("banks") if isinstance(obj, dict) else None
        if not isinstance(banks, dict):
            continue

        subtypes = sorted(list(banks.keys()))
        enabled_subtypes[super_type] = subtypes

        for sub_type in subtypes:
            # items per (super, sub, difficulty)
            items = load_en_bank_items(super_type, sub_type, diff)  # returns [{"w","ipa"}...]
            picked = _pick_n_from_items(items, n)
            if not picked:
                continue

            plan.append({
                "super_type": super_type,
                "sub_type": sub_type,
                "words": picked  # keep dicts so we can show ipa on UI if needed
            })
            total += len(picked)

    if total <= 0:
        raise HTTPException(status_code=500, detail="EN test banks empty / failed to build plan")

    flat_queue = build_test_flat_queue_en(plan)
    plan_obj = {
        "language": TEST_EN_LANG,
        "difficulty": diff,
        "words_per_subtype": n,
        "threshold": th,
        "plan": plan,
        "flatQueue": flat_queue,
        "total_questions": total,
    }

    config_obj = {
        "enabled_super_order": TEST_EN_SUPER_ORDER,
        "enabled_subtypes": enabled_subtypes,
        "bank_files": EN_BANK_FILES,
        "difficulty": diff,
    }

    # create tests row
    test_id = create_test_session_row_v2(
        user_id=current_user_id,
        test_name=f"SLP Screening (EN)",
        language=TEST_EN_LANG,
        words_per_subtype=n,
        threshold=th,
        plan_obj=plan_obj,
        total_questions=total,
        config_obj=config_obj,
    )

    return TestStartENResp(
        test_id=test_id,
        language=TEST_EN_LANG,
        words_per_subtype=n,
        threshold=th,
        plan=plan,
        total_questions=total,
    )

class TestAnalyzeENResp(BaseModel):
    test_record_id: str
    test_id: str
    super_type: str
    sub_type: str
    target_word: str
    target_ipa: str
    asr_word: str
    asr_ipa: str
    correct_rate: int
    status: str
    severity: Optional[str] = None
    check_results: dict

@app.post("/api/test_en/analyze", response_model=TestAnalyzeENResp)
async def api_test_en_analyze_real(
    file: UploadFile = File(...),
    test_id: str = Form(...),
    super_type: str = Form(...),
    sub_type: str = Form(...),
    target_word: str = Form(...),
    question_index: int = Form(...),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Analyze one EN test question:
    - verify test ownership + running
    - run English ASR + analyze_record_payload
    - save to test_records (structure same as CN test_records)
    - update tests.progress_cursor
    """
    if not test_id:
        raise HTTPException(status_code=400, detail="Missing test_id")

    super_type = _normalize_super_type_en(super_type)
    sub_type = (sub_type or "").strip()
    target_word = (target_word or "").strip()
    qidx = int(question_index)

    if super_type not in TEST_EN_SUPER_ORDER:
        raise HTTPException(status_code=400, detail="Invalid super_type (EN only supports Addition/Omission/Substitution)")
    if not sub_type:
        raise HTTPException(status_code=400, detail="Missing sub_type")
    if not target_word:
        raise HTTPException(status_code=400, detail="Missing target_word")

    # verify test row
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT patient_id, status, language, threshold, words_per_subtype, plan FROM tests WHERE id=?", (test_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="test_id not found")
    if row[0] != current_user_id:
        raise HTTPException(status_code=403, detail="Not allowed")
    if str(row[1] or "").lower() != "running":
        raise HTTPException(status_code=400, detail="Test is not running")
    if str(row[2] or "") != TEST_EN_LANG:
        raise HTTPException(status_code=400, detail="This test is not an EN test")

    # ensure (super_type, sub_type) exists in this test plan (avoid client spoof)
    plan_obj = {}
    try:
        plan_obj = json.loads(row[5]) if row[5] else {}
    except Exception:
        plan_obj = {}
    flatq = plan_obj.get("flatQueue") if isinstance(plan_obj, dict) else None
    if isinstance(flatq, list) and 0 <= qidx < len(flatq):
        expected = flatq[qidx] if isinstance(flatq[qidx], dict) else None
        if expected:
            exp_st = _normalize_super_type_en(expected.get("super_type"))
            exp_sb = str(expected.get("sub_type") or "").strip()
            exp_w = str(expected.get("word") or "").strip()
            if exp_st != super_type or exp_sb != sub_type or exp_w != target_word:
                raise HTTPException(status_code=400, detail="Submitted question does not match server plan")
    else:
        # if flatQueue missing, we skip strict match (but your start saves it, so normally won't happen)
        pass

    content = await file.read()

    # English ASR
    asr = transcribe_bytes_en_full(content, target_word=target_word)
    target_ipa = (asr.get("target_ipa") or "").strip()
    asr_ipa = (asr.get("asr_ipa") or "").strip()
    asr_word = (asr.get("asr_word") or "").strip() or "N/A"

    # keep consistency if needed
    asr_word, asr_ipa = enforce_word_ipa_consistency(target_word, target_ipa, asr_word, asr_ipa)

    # difficulty bucket (primary|secondary|advanced)
    diff = infer_en_difficulty_from_user_id(current_user_id)

    # analyze via your EN LLM pipeline
    req_id = str(uuid.uuid4())
    req_date = datetime.now().strftime("%Y-%m-%d")

    # IMPORTANT: analyze_record_payload expects test_type lower-case in your EN endpoints:
    # - addition/omission/substitution
    test_type_lower = super_type.lower()

    payload = analyze_record_payload(
        id=req_id,
        user_id=current_user_id,
        date=req_date,
        target_word=target_word,
        target_ipa=target_ipa,
        asr_word=asr_word,
        asr_ipa=asr_ipa,
        difficulty=diff,
        language="en",
        test_type=test_type_lower,
        category=sub_type,
        num_cases=None,
        max_new_tokens=None,
        temperature=0.2,
        top_p=1.0,
        repetition_penalty=1.05,
    )

    # Build test check_results: add meta for test
    check_results = payload.get("check_results") if isinstance(payload.get("check_results"), dict) else {}
    if not isinstance(check_results, dict):
        check_results = {}

    meta = check_results.get("meta") if isinstance(check_results.get("meta"), dict) else {}
    meta.update({
        "event": "test_analyze_en",
        "test_id": test_id,
        "language": TEST_EN_LANG,
        "question_index": qidx,
        "super_type": super_type,
        "sub_type": sub_type,
    })
    check_results["meta"] = meta

    # align fields
    correct_rate = int(payload.get("correct_rate") or 0)
    status = str(payload.get("status") or "")
    severity = payload.get("severity", None)
    score_points = payload.get("score", None)

    # ✅ Save into test_records + update tests cursor in ONE transaction
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id FROM test_records WHERE test_id=? AND question_index=? LIMIT 1",
            (test_id, int(qidx)),
        )
        exists = cur.fetchone()
        if exists:
            raise HTTPException(status_code=409, detail="This question was already submitted. Please click Next.")
        
        test_record_id = save_test_record_v2_tx(
            cur,
            test_id=test_id,
            question_index=qidx,
            user_id=current_user_id,
            language="English",
            test_type=super_type,     # ✅ test_type = super_type (as you requested)
            category=sub_type,        # ✅ category = sub_type
            target_word=target_word,
            target_ipa=target_ipa,
            asr_ipa=asr_ipa,
            asr_word=asr_word,
            correct_rate=correct_rate,
            check_results=check_results,
            difficulty_level=diff,    # normalize_difficulty will map, OK
            score=score_points if isinstance(score_points, int) else None,
            severity=severity if isinstance(severity, str) else None,
        )

        # recompute done count
        cur.execute("SELECT COUNT(DISTINCT question_index) FROM test_records WHERE test_id=?", (test_id,))
        done_count = int(cur.fetchone()[0] or 0)

        cur.execute("""
            UPDATE tests
            SET progress_cursor=?,
                current_question_index=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (done_count, done_count, test_id))

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return TestAnalyzeENResp(
        test_record_id=test_record_id,
        test_id=test_id,
        super_type=super_type,
        sub_type=sub_type,
        target_word=target_word,
        target_ipa=target_ipa,
        asr_word=asr_word,
        asr_ipa=asr_ipa,
        correct_rate=correct_rate,
        status=status,
        severity=severity if isinstance(severity, str) else None,
        check_results=check_results,
    )

class TestFinishENResp(BaseModel):
    test_id: str
    language: str
    threshold: int
    words_per_subtype: int
    confirmed: List[Dict[str, Any]]
    has_slp: int

@app.post("/api/test_en/finish", response_model=TestFinishENResp)
def api_test_en_finish_real(
    test_id: str = Query(...),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Finish EN test:
    - count final_label == 'slp' per (super_type, sub_type)
    - if count >= threshold => insert into slp_patient (language='en')
    - update tests to finished + result json
    """
    if not test_id:
        raise HTTPException(status_code=400, detail="Missing test_id")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT patient_id, language, threshold, words_per_subtype FROM tests WHERE id=?", (test_id,))
    trow = cur.fetchone()
    if not trow:
        conn.close()
        raise HTTPException(status_code=404, detail="test_id not found")
    if trow[0] != current_user_id:
        conn.close()
        raise HTTPException(status_code=403, detail="Not allowed")
    if str(trow[1] or "") != TEST_EN_LANG:
        conn.close()
        raise HTTPException(status_code=400, detail="This test is not an EN test")

    threshold = int(trow[2] or TEST_EN_SLP_THRESHOLD_DEFAULT)
    wps = int(trow[3] or TEST_EN_WORDS_PER_SUBTYPE_DEFAULT)

    # load all test_records
    cur.execute("""
        SELECT test_type, category, check_results
        FROM test_records
        WHERE test_id=?
        ORDER BY question_index ASC
    """, (test_id,))
    rows = cur.fetchall()
    conn.close()

    counter: dict[tuple[str, str], int] = {}

    for st, sb, cr_json in rows:
        st = _normalize_super_type_en(st)
        sb = str(sb or "").strip()

        try:
            obj = json.loads(cr_json) if cr_json else {}
        except Exception:
            obj = {}

        if not isinstance(obj, dict):
            continue

        meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
        # must match this test_id
        if str(meta.get("test_id") or "") != str(test_id):
            continue

        # get detector result from payload
        det_key = _detector_key_for_super_en(st)
        det = obj.get(det_key) if det_key else None
        if not isinstance(det, dict):
            continue

        final_label = str(det.get("final_label") or "").strip().lower()
        if final_label == "slp":
            counter[(st, sb)] = counter.get((st, sb), 0) + 1

    confirmed = []
    confirmed_pairs: list[tuple[str, str]] = []

    for (st, sb), c in sorted(counter.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        if c >= threshold:
            confirmed.append({"super_type": st, "sub_type": sb, "slp_count": c})
            confirmed_pairs.append((st, sb))

    # insert slp_patient rows
    insert_slp_patient_rows(
        user_id=current_user_id,
        test_id=test_id,
        confirmed=confirmed_pairs,
        language=TEST_EN_LANG,
    )

    has_slp = 1 if confirmed_pairs else 0

    summary_obj = {
        "language": TEST_EN_LANG,
        "threshold": threshold,
        "words_per_subtype": wps,
        "confirmed": confirmed,
        "counts": {f"{k[0]}::{k[1]}": v for k, v in counter.items()},
    }

    update_test_session_result(test_id=test_id, has_slp=has_slp, result_obj=summary_obj)

    return TestFinishENResp(
        test_id=test_id,
        language=TEST_EN_LANG,
        threshold=threshold,
        words_per_subtype=wps,
        confirmed=confirmed,
        has_slp=has_slp,
    )

class TestResumeENResp(BaseModel):
    available: bool
    test_id: Optional[str] = None
    status: Optional[str] = None
    progress_cursor: int = 0
    plan: Optional[dict] = None

@app.get("/api/test_en/resume", response_model=TestResumeENResp)
def api_test_en_resume_real(current_user_id: str = Depends(get_current_user_id)):
    """
    Resume latest running EN test for this user.
    """
    ensure_tests_table_v2_columns()

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, status, progress_cursor, plan
            FROM tests
            WHERE patient_id=?
              AND status='running'
              AND language=?
            ORDER BY datetime(updated_at) DESC, datetime(date) DESC
            LIMIT 1
        """, (current_user_id, TEST_EN_LANG))
        row = cur.fetchone()

        if not row:
            return TestResumeENResp(available=False)

        test_id, status, progress_cursor, plan_json = row
        plan_obj = None
        if plan_json:
            try:
                plan_obj = json.loads(plan_json)
            except Exception:
                plan_obj = {"raw": plan_json}

        return TestResumeENResp(
            available=True,
            test_id=test_id,
            status=status,
            progress_cursor=int(progress_cursor or 0),
            plan=plan_obj,
        )
    finally:
        conn.close()

# ---- Staff: patient picker (only patients this SLP has sessions with) ----
class StaffPatientItem(BaseModel):
    id: str
    username: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    language: Optional[str] = None
    last_session_at: Optional[str] = None

@app.get("/api/slp/patients", response_model=dict)
def api_slp_patients(
    payload: dict = Depends(get_current_staff_payload),
):
    ensure_customer_service_tables()

    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    slp_id = payload["sub"]

    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # patients that have at least one session with this slp
        cur.execute("""
            SELECT
              p.id, p.username, p.name, p.email, p.language,
              MAX(COALESCE(s.last_message_at, s.created_at)) AS last_session_at
            FROM service_sessions s
            JOIN patients p ON p.id = s.patient_id
            WHERE s.slp_id=?
              AND s.status IN ('assigned','open','closed')   -- "talked to" scope
            GROUP BY p.id
            ORDER BY datetime(last_session_at) DESC
        """, (slp_id,))
        items = [dict(r) for r in cur.fetchall()]
        return {"items": items}
    finally:
        conn.close()

@app.get("/api/slp/patient/{patient_id}/records_analysis")
def api_slp_patient_records_analysis(
    patient_id: str,
    language: Optional[str] = Query(None, description="Chinese|English (optional)"),
    limit: int = Query(500, ge=1, le=2000),
    payload: dict = Depends(get_current_staff_payload),
):
    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")
    ensure_customer_service_tables()

    slp_id = payload["sub"]

    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # permission: must have a session with this patient
        cur.execute("""
            SELECT 1
            FROM service_sessions
            WHERE patient_id=? AND slp_id=? AND status IN ('assigned','open','closed')
            LIMIT 1
        """, (patient_id, slp_id))
        if not cur.fetchone():
            raise HTTPException(status_code=403, detail="not allowed")

        where_lang = ""
        params = [patient_id]

        if language and str(language).strip() != "":
            where_lang = " AND language=?"
            params.append(str(language).strip())

        params.append(int(limit))

        cur.execute(f"""
            SELECT *
            FROM records_analysis
            WHERE user_id=?
            {where_lang}
            ORDER BY datetime(created_at) DESC
            LIMIT ?
        """, tuple(params))

        return {"records": [dict(r) for r in cur.fetchall()]}
    finally:
        conn.close()

@app.get("/api/slp/patient/{patient_id}/tests")
def api_slp_patient_tests(
    patient_id: str,
    language: Optional[str] = Query(None, description="cn|en (optional)"),
    limit: int = Query(200, ge=1, le=500),
    payload: dict = Depends(get_current_staff_payload),
):
    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")
    ensure_customer_service_tables()

    slp_id = payload["sub"]
    lang = None
    if language is not None and str(language).strip() != "":
        lang = normalize_practice_language(language)

    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 1
            FROM service_sessions
            WHERE patient_id=? AND slp_id=? AND status IN ('assigned','open','closed')
            LIMIT 1
        """, (patient_id, slp_id))
        if not cur.fetchone():
            raise HTTPException(status_code=403, detail="not allowed")

        where_lang = ""
        params = [patient_id]
        if lang:
            where_lang = " AND language=?"
            params.append(lang)
        params.append(int(limit))

        cur.execute(f"""
            SELECT
              id, test_name, status, language,
              threshold, words_per_subtype,
              progress_cursor, total_questions, current_question_index,
              has_slp, date, updated_at, finished_at
            FROM tests
            WHERE patient_id=?
              AND status <> 'cancelled'
              {where_lang}
            ORDER BY datetime(updated_at) DESC, datetime(date) DESC
            LIMIT ?
        """, tuple(params))

        return {"items": [dict(r) for r in cur.fetchall()]}
    finally:
        conn.close()


@app.get("/api/slp/patient/{patient_id}/tests/{test_id}")
def api_slp_patient_test_detail(
    patient_id: str,
    test_id: str,
    payload: dict = Depends(get_current_staff_payload),
):
    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")
    ensure_customer_service_tables()

    slp_id = payload["sub"]

    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 1
            FROM service_sessions
            WHERE patient_id=? AND slp_id=? AND status IN ('assigned','open','closed')
            LIMIT 1
        """, (patient_id, slp_id))
        if not cur.fetchone():
            raise HTTPException(status_code=403, detail="not allowed")

        cur.execute("""
            SELECT *
            FROM tests
            WHERE id=? AND patient_id=? AND status <> 'cancelled'
            LIMIT 1
        """, (test_id, patient_id))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="test not found")

        # return as patient endpoint shape
        def try_json(s):
            if not s:
                return None
            try:
                return json.loads(s)
            except Exception:
                return {"raw": s}

        return {
            "test": {
                "id": row["id"],
                "test_name": row["test_name"],
                "status": row["status"],
                "progress_cursor": int(row["progress_cursor"] or 0),
                "total_questions": int(row["total_questions"] or 0),
                "current_question_index": int(row["current_question_index"] or 0),
                "plan": try_json(row["plan"]),
                "config": try_json(row["config_json"]),
                "threshold": int(row["threshold"] or 0),
                "words_per_subtype": int(row["words_per_subtype"] or 0),
                "language": row["language"],
                "has_slp": int(row["has_slp"] or 0),
                "result": try_json(row["result"]),
                "created_at": row["date"],
                "updated_at": row["updated_at"],
                "finished_at": row["finished_at"],
            }
        }
    finally:
        conn.close()


@app.get("/api/slp/patient/{patient_id}/tests/{test_id}/records")
def api_slp_patient_test_records(
    patient_id: str,
    test_id: str,
    payload: dict = Depends(get_current_staff_payload),
):
    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")
    ensure_customer_service_tables()

    slp_id = payload["sub"]

    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 1
            FROM service_sessions
            WHERE patient_id=? AND slp_id=? AND status IN ('assigned','open','closed')
            LIMIT 1
        """, (patient_id, slp_id))
        if not cur.fetchone():
            raise HTTPException(status_code=403, detail="not allowed")

        cur.execute("SELECT patient_id, status FROM tests WHERE id=? LIMIT 1", (test_id,))
        t = cur.fetchone()
        if not t or t["patient_id"] != patient_id or str(t["status"]).lower() == "cancelled":
            raise HTTPException(status_code=404, detail="test not found")

        cur.execute("""
            SELECT *
            FROM test_records
            WHERE test_id=?
            ORDER BY question_index ASC, datetime(created_at) ASC
        """, (test_id,))
        return {"items": [dict(r) for r in cur.fetchall()]}
    finally:
        conn.close()


@app.get("/api/slp/patient/{patient_id}/tests/{test_id}/slp")
def api_slp_patient_test_slp(
    patient_id: str,
    test_id: str,
    language: Optional[str] = Query(None, description="cn|en (optional)"),
    payload: dict = Depends(get_current_staff_payload),
):
    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")
    ensure_customer_service_tables()

    slp_id = payload["sub"]
    lang = None
    if language is not None and str(language).strip() != "":
        lang = normalize_practice_language(language)

    conn = get_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 1
            FROM service_sessions
            WHERE patient_id=? AND slp_id=? AND status IN ('assigned','open','closed')
            LIMIT 1
        """, (patient_id, slp_id))
        if not cur.fetchone():
            raise HTTPException(status_code=403, detail="not allowed")

        where_lang = ""
        params = [test_id]
        if lang:
            where_lang = " AND language=?"
            params.append(lang)

        cur.execute(f"""
            SELECT *
            FROM slp_patient
            WHERE test_id=?
            {where_lang}
            ORDER BY datetime(created_at) DESC, datetime(datetime) DESC
        """, tuple(params))

        return {"items": [dict(r) for r in cur.fetchall()]}
    finally:
        conn.close()

@app.get("/api/slp/patient/{patient_id}/training/tasks")
def api_slp_patient_training_tasks(
    patient_id: str,
    payload: dict = Depends(get_current_staff_payload),
):
    """
    SLP 查看某个病人的 training tasks（CN/EN + day1..7）。
    权限：该 SLP 必须与该病人存在 service_sessions (assigned/open/closed)。
    返回格式对齐 /api/tasks/training/all-active
    """
    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")
    ensure_customer_service_tables()

    slp_id = payload["sub"]

    conn = get_db()
    cur = conn.cursor()
    try:
        # permission check
        cur.execute("""
            SELECT 1
            FROM service_sessions
            WHERE patient_id=? AND slp_id=? AND status IN ('assigned','open','closed')
            LIMIT 1
        """, (patient_id, slp_id))
        if not cur.fetchone():
            raise HTTPException(status_code=403, detail="not allowed")

        # latest allocated_at for this patient (training tasks only)
        cur.execute("""
            SELECT allocated_at
            FROM tasks
            WHERE user_id=?
              AND is_active=1
              AND allocated_at IS NOT NULL
              AND super_type IS NOT NULL
              AND sub_type IS NOT NULL
            ORDER BY datetime(allocated_at) DESC
            LIMIT 1
        """, (patient_id,))
        r = cur.fetchone()
        latest_alloc = r[0] if r else None
        latest_allowed = _task_allowed_day_index(latest_alloc) if latest_alloc else None

        # all active training tasks
        cur.execute("""
            SELECT
              id, super_type, sub_type, task_kind, language,
              allocated_at, day_index, total_times, progress_times, status, is_active,
              reward_claimed, reward_claimed_at
            FROM tasks
            WHERE user_id=?
              AND is_active=1
              AND super_type IS NOT NULL
              AND sub_type IS NOT NULL
            ORDER BY
              datetime(allocated_at) DESC,
              language ASC,
              super_type ASC,
              sub_type ASC,
              task_kind ASC,
              day_index ASC
        """, (patient_id,))
        rows = cur.fetchall()

        tasks = []
        for row in rows:
            (tid, st, sb, kind, lang, alloc, day_idx, total, prog, status, is_active, reward_claimed, reward_claimed_at) = row
            tasks.append({
                "id": tid,
                "super_type": st,
                "sub_type": sb,
                "task_kind": str(kind or "vocab"),
                "language": str(lang or "cn"),
                "allocated_at": alloc,
                "day_index": int(day_idx or 1),
                "total_times": int(total or 0),
                "progress_times": int(prog or 0),
                "status": str(status or "pending"),
                "is_active": int(is_active or 0),
                "reward_claimed": int(reward_claimed or 0),
                "reward_claimed_at": reward_claimed_at,
            })

        return {
            "available": bool(tasks),
            "latest_allocated_at": latest_alloc,
            "latest_allowed_day_index": latest_allowed,
            "tasks": tasks,
            "reason": None if tasks else "NO_ACTIVE_TRAINING_TASKS",
        }
    finally:
        conn.close()

@app.get("/debug_g2p")
async def debug_g2p(word: str = "RIDBON"):
    # Show exactly which file is running + what espeak returns inside the server process
    info = {
        "module": __name__,
        "file": str(Path(__file__).resolve()),
        "word": word,
    }

    # Check espeak availability
    try:
        ver = subprocess.check_output(
            ["espeak", "--version"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
        ).strip()
        info["espeak_version"] = ver
    except Exception as e:
        info["espeak_version_error"] = repr(e)

    # IPA output
    try:
        ipa_out = subprocess.check_output(
            ["espeak", "-q", "-v", "en-us", "--ipa=3", word],
            text=True,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
        ).strip().replace("\u200d", "")
        info["espeak_ipa_raw"] = ipa_out
        info["espeak_ipa_normalized"] = normalize_ipa(ipa_out)
    except Exception as e:
        info["espeak_ipa_error"] = repr(e)

    # -x fallback output
    try:
        x_out = subprocess.check_output(
            ["espeak", "-q", "-v", "en-us", "-x", word],
            text=True,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
        ).strip()
        info["espeak_x"] = x_out
    except Exception as e:
        info["espeak_x_error"] = repr(e)

    # What YOUR pipeline would return
    try:
        info["word_to_ipa_or_star"] = word_to_ipa_or_star(word)
    except Exception as e:
        info["word_to_ipa_or_star_error"] = repr(e)

    return JSONResponse(info)

# ==============================
# ✅ SLP Task Editor: Update Patient Training Tasks (CN/EN separate)
# ==============================
from pydantic import BaseModel, Field

class SlpTaskEditItem(BaseModel):
    super_type: str
    sub_type: str
    total_times: int = Field(..., ge=1, le=9999)

class SlpTaskEditDay(BaseModel):
    day_index: int = Field(..., ge=1, le=7)
    items: List[SlpTaskEditItem] = []

class SlpTaskEditReq(BaseModel):
    patient_id: str
    language: str = "cn"        # "cn" | "en"
    task_kind: str = "vocab"    # "vocab" | "sentence"
    days: List[SlpTaskEditDay]
    meta: Optional[dict] = None

def _normalize_task_lang(lang: str | None) -> str:
    s = (lang or "").strip().lower()
    if s in ("cn", "zh", "zh-cn", "chinese"):
        return "cn"
    if s in ("en", "english"):
        return "en"
    return "cn"

def _normalize_task_kind(kind: str | None) -> str:
    k = (kind or "vocab").strip().lower()
    if k not in ("vocab", "sentence"):
        return "vocab"
    return k

@app.post("/api/slp/patient/{patient_id}/training/tasks/update")
def api_slp_update_patient_training_tasks(
    patient_id: str,
    data: SlpTaskEditReq,
    payload: dict = Depends(get_current_staff_payload),
):
    """
    SLP 覆寫「某病人」某語言(cn/en) 的「最新一批 allocated_at」 training tasks（day1..7）。
    - 不更新 allocated_at（沿用該 batch 最新值）
    - 不真的刪 row：把不在新配置內的舊 task 設 is_active=0（soft delete）
    - 新增：INSERT 新 row（allocated_at 同 batch）
    - 更新：存在則 UPDATE total_times（progress_times 保留不動）
    """
    ensure_task_table_v2()
    ensure_customer_service_tables()

    # ---- auth: only slp ----
    if payload.get("role") != "slp":
        raise HTTPException(status_code=403, detail="Only SLP")

    slp_id = payload.get("sub")
    if not slp_id:
        raise HTTPException(status_code=401, detail="Invalid staff token")

    # ---- path/body patient_id must match ----
    if str(patient_id) != str(data.patient_id):
        raise HTTPException(status_code=400, detail="patient_id mismatch")

    # ---- permission: slp must have sessions with patient ----
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 1
            FROM service_sessions
            WHERE patient_id=? AND slp_id=? AND status IN ('assigned','open','closed')
            LIMIT 1
        """, (patient_id, slp_id))
        if not cur.fetchone():
            raise HTTPException(status_code=403, detail="not allowed")

        lang = _normalize_task_lang(data.language)
        task_kind = _normalize_task_kind(data.task_kind)

        # ---- validate days: must cover 1..7 (can be empty items) ----
        days_map: dict[int, list[SlpTaskEditItem]] = {}
        for d in (data.days or []):
            di = int(d.day_index)
            if di < 1 or di > 7:
                raise HTTPException(status_code=400, detail="Invalid day_index")
            days_map[di] = d.items or []

        for di in range(1, 8):
            if di not in days_map:
                # require explicit day entries (so update is deterministic)
                raise HTTPException(status_code=400, detail=f"Missing day_index={di}")

        # ---- find latest allocated_at batch for this patient+lang+kind ----
        cur.execute("""
            SELECT allocated_at
            FROM tasks
            WHERE user_id=?
              AND is_active=1
              AND allocated_at IS NOT NULL
              AND language=?
              AND task_kind=?
              AND super_type IS NOT NULL
              AND sub_type IS NOT NULL
            ORDER BY datetime(allocated_at) DESC
            LIMIT 1
        """, (patient_id, lang, task_kind))
        r = cur.fetchone()
        if not r or not r[0]:
            raise HTTPException(status_code=400, detail="NO_ACTIVE_BATCH")

        allocated_at = r[0]

        # ---- load existing tasks in this batch (for lang+kind only) ----
        cur.execute("""
            SELECT id, day_index, super_type, sub_type, total_times, progress_times, status, is_active
            FROM tasks
            WHERE user_id=?
              AND language=?
              AND task_kind=?
              AND is_active=1
              AND allocated_at IS NOT NULL
              AND datetime(allocated_at)=datetime(?)
              AND day_index BETWEEN 1 AND 7
              AND super_type IS NOT NULL
              AND sub_type IS NOT NULL
        """, (patient_id, lang, task_kind, allocated_at))
        rows = cur.fetchall()

        # key: (day_index, super_type, sub_type) -> row
        existing = {}
        for row in rows:
            tid, day_index, st, sb, total_times, progress_times, status, is_active = row
            existing[(int(day_index), str(st), str(sb))] = {
                "id": tid,
                "total_times": int(total_times or 0),
                "progress_times": int(progress_times or 0),
                "status": str(status or "pending"),
                "is_active": int(is_active or 0),
            }

        # desired keys set
        desired_keys = set()
        for di in range(1, 8):
            for it in (days_map.get(di) or []):
                st = str(it.super_type or "").strip()
                sb = str(it.sub_type or "").strip()
                if not st or not sb:
                    raise HTTPException(status_code=400, detail="Empty super_type/sub_type")
                desired_keys.add((di, st, sb))

        # ---- soft delete: existing but not desired ----
        for k, ex in existing.items():
            if k not in desired_keys:
                cur.execute("UPDATE tasks SET is_active=0 WHERE id=?", (ex["id"],))

        # ---- upsert desired ----
        upserted = {"inserted": 0, "updated": 0}
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for di in range(1, 8):
            for it in (days_map.get(di) or []):
                st = str(it.super_type).strip()
                sb = str(it.sub_type).strip()
                total_times = int(it.total_times)

                k = (di, st, sb)
                if k in existing:
                    # UPDATE only total_times (keep progress/status as-is)
                    cur.execute("""
                        UPDATE tasks
                        SET total_times=?,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE id=?
                    """, (total_times, existing[k]["id"]))
                    upserted["updated"] += 1
                else:
                    # INSERT new row into same batch allocated_at
                    # note: title/category are required by schema, provide defaults
                    tid = str(uuid.uuid4())
                    cur.execute("""
                        INSERT INTO tasks
                        (id, title, description, category, due_date,
                         total_times, progress_times, integral, status, created_at,
                         user_id, super_type, sub_type, language, allocated_at,
                         valid_days, day_index, is_active, task_kind, task_key,
                         reward_claimed, reward_claimed_at)
                        VALUES
                        (?, ?, ?, 'daily', NULL,
                         ?, 0, 0, 'pending', CURRENT_TIMESTAMP,
                         ?, ?, ?, ?, ?,
                         7, ?, 1, ?, NULL,
                         0, NULL)
                    """, (
                        tid,
                        f"SLP Edited Task ({lang})",
                        f"Edited by SLP {slp_id} at {now}",
                        total_times,
                        patient_id,
                        st,
                        sb,
                        lang,
                        allocated_at,
                        int(di),
                        task_kind,
                    ))
                    upserted["inserted"] += 1

        # ---- audit log ----
        try:
            audit_log(
                actor_type="slp",
                actor_id=slp_id,
                action="SLP_EDIT_PATIENT_TASKS",
                resource_type="tasks_batch",
                resource_id=str(allocated_at),
                metadata=json.dumps({
                    "patient_id": patient_id,
                    "language": lang,
                    "task_kind": task_kind,
                    "inserted": upserted["inserted"],
                    "updated": upserted["updated"],
                    "meta": data.meta or {},
                }, ensure_ascii=False),
            )
        except Exception:
            pass

        conn.commit()

        # return refreshed batch for frontend to re-render
        cur.execute("""
            SELECT
              id, super_type, sub_type, task_kind, language,
              allocated_at, day_index, total_times, progress_times, status, is_active,
              reward_claimed, reward_claimed_at
            FROM tasks
            WHERE user_id=?
              AND language=?
              AND task_kind=?
              AND allocated_at IS NOT NULL
              AND datetime(allocated_at)=datetime(?)
              AND day_index BETWEEN 1 AND 7
            ORDER BY day_index ASC, super_type ASC, sub_type ASC
        """, (patient_id, lang, task_kind, allocated_at))
        out_rows = cur.fetchall()

        tasks = []
        for row in out_rows:
            (tid, st, sb, kind, lg, alloc, day_idx,
             total, prog, status, is_active, reward_claimed, reward_claimed_at) = row
            tasks.append({
                "id": tid,
                "super_type": st,
                "sub_type": sb,
                "task_kind": str(kind or "vocab"),
                "language": str(lg or lang),
                "allocated_at": alloc,
                "day_index": int(day_idx or 1),
                "total_times": int(total or 0),
                "progress_times": int(prog or 0),
                "status": str(status or "pending"),
                "is_active": int(is_active or 0),
                "reward_claimed": int(reward_claimed or 0),
                "reward_claimed_at": reward_claimed_at,
            })

        allowed = _task_allowed_day_index(allocated_at)

        return {
            "msg": "ok",
            "patient_id": patient_id,
            "language": lang,
            "task_kind": task_kind,
            "allocated_at": allocated_at,
            "allowed_day_index": allowed,
            "upserted": upserted,
            "tasks": tasks,
        }

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"update tasks failed: {str(e)}")
    finally:
        conn.close()

from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse

FRONTEND_PUBLIC_DIR = (BASE_DIR / ".." / "frontend" / "public").resolve()
UNITY_GAME1_DIR = (FRONTEND_PUBLIC_DIR / "Game1").resolve()

# Basic MIME map for Unity files
_MIME_MAP = {
    ".js": "application/javascript",
    ".wasm": "application/wasm",
    ".data": "application/octet-stream",
    ".json": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".mp4": "video/mp4",
}

def _guess_mime_for_unity_file(p: Path) -> str:
    """
    For *.framework.js.br we want JS mime.
    For *.wasm.br we want wasm mime.
    For *.data.br we want octet-stream.
    Strategy:
    - If file endswith .br => look at the suffix BEFORE .br
    - else => look at normal suffix
    """
    suffixes = p.suffixes  # e.g. ['.framework', '.js', '.br']
    if suffixes and suffixes[-1] == ".br":
        # use the suffix before .br if available
        if len(suffixes) >= 2:
            ext = suffixes[-2]  # '.js' or '.wasm' or '.data'
        else:
            ext = ".bin"
    else:
        ext = p.suffix or ".bin"

    return _MIME_MAP.get(ext.lower(), "application/octet-stream")

@app.get("/unity/Game1/{file_path:path}")
def unity_game1_files(file_path: str):
    """
    Serve files under frontend/public/Game1/...
    Example:
      /unity/Game1/Build/Game1.loader.js
      /unity/Game1/Build/Game1.wasm.br
    """
    # Prevent path traversal
    abs_path = (UNITY_GAME1_DIR / file_path).resolve()
    if not str(abs_path).startswith(str(UNITY_GAME1_DIR)):
        raise HTTPException(status_code=403, detail="Forbidden path")

    if not abs_path.exists() or not abs_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    headers = {
        "Cache-Control": "no-cache",
        "Cross-Origin-Resource-Policy": "cross-origin",
    }

    # Only .br files get Content-Encoding: br
    if abs_path.suffix == ".br":
        headers["Content-Encoding"] = "br"

    return FileResponse(
        path=str(abs_path),
        media_type=_guess_mime_for_unity_file(abs_path),
        headers=headers,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)