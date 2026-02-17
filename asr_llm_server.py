from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn, os, tempfile, json, re
import torch, librosa
from transformers import Wav2Vec2Processor, AutoModelForCTC
from datetime import datetime
import uuid
from pathlib import Path
from phonemizer import phonemize
import eng_to_ipa as ipa
import logging
from pathlib import Path
import subprocess

logger = logging.getLogger("asr.en")

logger.warning("### ENGLISH_US ASR SERVER VERSION: star-fix enabled ###")
logger.warning("[IMPORT DEBUG] English_US.asr_llm_server file = %s", str(Path(__file__).resolve()))
logger.warning("[IMPORT DEBUG] module = %s", __name__)

# ======================================================
# 0. G2P (better fallback) + acronym spell-out
# ======================================================
LETTER_IPA = {
    "A": "eɪ", "B": "biː", "C": "siː", "D": "diː", "E": "iː", "F": "ɛf", "G": "dʒiː",
    "H": "eɪtʃ", "I": "aɪ", "J": "dʒeɪ", "K": "keɪ", "L": "ɛl", "M": "ɛm", "N": "ɛn",
    "O": "oʊ", "P": "piː", "Q": "kjuː", "R": "ɑːr", "S": "ɛs", "T": "tiː", "U": "juː",
    "V": "viː", "W": "ˈdʌbəl.juː", "X": "ɛks", "Y": "waɪ", "Z": "ziː"
}

_VOWELS = set("AEIOU")

def looks_like_acronym(token: str) -> bool:
    """
    改良版：避免 WARE/BEATS 呢種普通全大寫詞被當 acronym。
    只把「全大寫、2-10字母、而且幾乎冇母音」當 acronym（例如 TMASE/USB/CNN）。
    """
    t = (token or "").strip()
    if not (t.isalpha() and t.isupper() and 2 <= len(t) <= 10):
        return False
    # 如果有母音，就更似普通單詞，唔 spell-out
    has_vowel = any(ch in _VOWELS for ch in t)
    return not has_vowel

def spell_out_ipa(token: str) -> str:
    # TMASE -> tiː ɛm eɪ ɛs iː
    return " ".join([LETTER_IPA.get(ch, "") for ch in token if ch.isalpha()]).strip()



def g2p_espeak_cli(word: str) -> str:
    w = (word or "").strip()
    if not w:
        return ""

    # 1) Prefer IPA output
    try:
        ipa = subprocess.check_output(
            ["espeak", "-q", "-v", "en-us", "--ipa=3", w],
            text=True,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
        ).strip()

        # Remove zero-width joiner if present (it can confuse downstream)
        ipa = ipa.replace("\u200d", "")

        if ipa:
            return ipa
    except Exception as e:
        logger.warning("[G2P-CLI] --ipa failed token=%r err=%r", w, e)

    # 2) Fallback to ASCII phoneme mnemonics
    try:
        x = subprocess.check_output(
            ["espeak", "-q", "-v", "en-us", "-x", w],
            text=True,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
        ).strip()
        return x
    except Exception as e:
        logger.warning("[G2P-CLI] -x failed token=%r err=%r", w, e)
        return ""
    
def g2p_espeak(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""

    last_err = None
    for backend in ("espeak-ng", "espeak"):
        try:
            out = phonemize(
                s.lower(),
                language="en-us",
                backend=backend,
                strip=True,
                preserve_punctuation=False,
                with_stress=True,
                njobs=1,
            )
            out = (out or "").replace("\n", " ").strip()
            logger.warning("[G2P] backend=%s token=%r out=%r", backend, s, out)
            if out:
                return out
        except Exception as e:
            last_err = e
            logger.warning("[G2P] backend=%s token=%r EXCEPTION=%r", backend, s, e)

    return ""

# ======================================================
# 1. 初始化 ASR 模型 (Wav2Vec2 CTC)
# ======================================================
HERE = Path(__file__).resolve().parent
ASR_MODEL_DIR = (HERE / "./../wav2vec2-large-960h").resolve()

if not ASR_MODEL_DIR.exists():
    raise RuntimeError(f"Local ASR model dir not found: {ASR_MODEL_DIR}")

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained(str(ASR_MODEL_DIR), local_files_only=True)
model = AutoModelForCTC.from_pretrained(str(ASR_MODEL_DIR), local_files_only=True)
model.to(device)
model.eval()

# ======================================================
# 2. 載入發音字典（target IPA / ASR word IPA 都可用）
#    新格式: { "mop": "/mɑp/", "green": "/ˈɡɹin/", ... }
# ======================================================
PRON_DICT_PATH = "./LLM/target_wordlist.json"
with open(PRON_DICT_PATH, "r", encoding="utf-8") as f:
    pronunciation_dict = json.load(f)

# ======================================================
# 3. IPA / text helpers
# ======================================================
_word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def normalize_ipa(ipa_str: str) -> str:
    # 去重音、去長音；保留你字典里的 /.../ 也没问题
    return (ipa_str or "").replace("ˈ", "").replace("ˌ", "").replace("ː", "").strip()

def dict_lookup_ipa(word: str) -> str:
    w = (word or "").lower().strip()
    if not w:
        return ""

    v = pronunciation_dict.get(w)
    if not v:
        return ""

    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list) and v:
        return str(v[0] or "").strip()

    return ""

def tokenize_words(text: str) -> list[str]:
    s = (text or "").strip()
    if not s:
        return []
    return _word_re.findall(s)

def join_tokens_as_word(tokens: list[str]) -> str:
    # 只黐字母 token：["WARE","BEATS"] -> "WAREBEATS"
    return "".join([t for t in (tokens or []) if t and t.isalpha()])

def join_ipa_tokens_no_space(ipa_tokens: list[str]) -> str:
    # 去掉 IPA token 內的空格，再黐埋
    # e.g. "tiː ɛm eɪ" -> "tiːɛmeɪ"
    compact = []
    for x in ipa_tokens or []:
        if not x:
            continue
        compact.append(x.replace(" ", ""))
    return "".join(compact)

def word_to_ipa_or_star(token: str) -> str:
    w = (token or "").strip()
    if not w:
        return ""

    logger.warning("[IPA] token=%r (enter)", w)

    # 1) acronym spell-out
    if looks_like_acronym(w):
        spelled = spell_out_ipa(w)
        logger.warning("[IPA] token=%r looks_like_acronym=True spelled=%r", w, spelled)
        if spelled:
            return normalize_ipa(spelled)

    # 2) dict
    ipa0 = dict_lookup_ipa(w)
    logger.warning("[IPA] token=%r dict=%r", w, ipa0)
    if ipa0:
        return normalize_ipa(ipa0)

    # 3) espeak CLI
    g2p0 = g2p_espeak_cli(w)
    logger.warning("[IPA] token=%r g2p_espeak_cli=%r", w, g2p0)
    if g2p0:
        out = normalize_ipa(g2p0)
        logger.warning("[IPA] token=%r normalized=%r", w, out)
        if out:
            return out

    # 4) last fallback: eng_to_ipa
    try:
        out = ipa.convert(w)
        out = normalize_ipa(out)
        logger.warning("[IPA] token=%r eng_to_ipa=%r", w, out)
        if out:
            return out
    except Exception as e:
        logger.warning("[IPA] token=%r eng_to_ipa EXC=%r", w, e)

    logger.warning("[IPA] token=%r RETURN_STAR", w)
    return f"{w.lower()}*"

def text_to_ipa_all_words(text: str) -> dict:
    tokens = tokenize_words(text)

    # Keep one output per token (IPA or token*)
    ipa_tokens = [word_to_ipa_or_star(w) for w in tokens]

    # ✅ Keep everything (including starred unknowns) in ipa_text
    ipa_text = " ".join([x for x in ipa_tokens if x])

    # For joined_ipa: only join *real IPA* tokens; skip starred ones
    ipa_tokens_for_join = [x for x in ipa_tokens if x and not x.endswith("*")]
    joined_ipa = join_ipa_tokens_no_space(ipa_tokens_for_join)

    return {
        "tokens": tokens,
        "ipa_tokens": ipa_tokens,
        "ipa_text": ipa_text,
        "joined_word": join_tokens_as_word(tokens),
        "joined_ipa": joined_ipa,
    }

def get_target_ipa(target_word: str) -> str:
    return text_to_ipa_all_words(target_word).get("ipa_text", "")

# ======================================================
# 4. ASR helpers
# ======================================================
def check_file(file: UploadFile):
    if not (file.filename or "").lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are allowed")
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"]:
        raise HTTPException(status_code=400, detail="Invalid MIME type")
    return True

def load_and_denoise(path: str) -> torch.Tensor:
    # 只做 resample + mono；你已經移除 DeepFilterNet
    audio, _ = librosa.load(path, sr=16000, mono=True)
    return torch.tensor(audio, dtype=torch.float32)  # (T,)

def recognize_word(audio_tensor: torch.Tensor) -> str:
    wav = audio_tensor.detach().cpu().float().flatten().numpy()

    inputs = processor(
        wav,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values=input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)

    text = processor.batch_decode(pred_ids)[0]
    return (text or "").strip()

# ======================================================
# 5. Core analyze function (shared by all routes)
# ======================================================
async def _analyze_core(file: UploadFile, target_word: str, difficulty_level: str):
    check_file(file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        audio = load_and_denoise(tmp_path)
        asr_text = recognize_word(audio)

        target_ipa = get_target_ipa(target_word)

        asr_ipa_pack = text_to_ipa_all_words(asr_text) if asr_text else {
            "tokens": [],
            "ipa_tokens": [],
            "ipa_text": "",
            "joined_word": "",
            "joined_ipa": "",
        }

        # ✅ You said: merge tokens to one word, then lowercase
        # e.g. "WE BOND" -> "WEBOND" -> "webond"
        asr_word = (asr_ipa_pack.get("joined_word") or "").lower()

        return JSONResponse({
            "id": str(uuid.uuid4()),
            "user_id": None,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "target_word": target_word,
            "target_ipa": target_ipa,

            # 原本欄位（兼容）
            "asr_word": asr_word,                 # ✅ now merged+lowercase
            "asr_ipa": asr_ipa_pack["ipa_text"],
            "asr_text": asr_text,

            # ✅ 新增：黐埋做一個 word / IPA 無空格
            "asr_joined_word": asr_ipa_pack["joined_word"],  # original case join
            "asr_joined_ipa": asr_ipa_pack["joined_ipa"],

            # full detail for ALL ASR words
            "asr_tokens": asr_ipa_pack["tokens"],
            "asr_ipa_tokens": asr_ipa_pack["ipa_tokens"],

            "difficulty_level": difficulty_level,
        })
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ======================================================
# 6. API routes
# ======================================================
app = FastAPI(title="ASR Pronunciation Analyzer (Joined-word + IPA no-space)")

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    target_word: str = Form(...),
    difficulty_level: str = Form("primary_school")
):
    return await _analyze_core(file=file, target_word=target_word, difficulty_level=difficulty_level)

# 兼容旧路由：用同一個 core，唔需要重複寫多次
_COMPAT_ENDPOINTS = [
    "/analyze_addition_words",
    "/analyze_substitution_words",
    "/analyze_omission_words",
    "/analyze_distortion_words",
]

def _make_compat_handler():
    async def handler(
        file: UploadFile = File(...),
        target_word: str = Form(...),
        difficulty_level: str = Form("primary_school")
    ):
        return await _analyze_core(file=file, target_word=target_word, difficulty_level=difficulty_level)
    return handler

for _path in _COMPAT_ENDPOINTS:
    app.post(_path)(_make_compat_handler())

# 给 main import 用
def transcribe_bytes_en(audio_bytes: bytes, target_word: str) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio = load_and_denoise(tmp_path)
        asr_text = recognize_word(audio)
        asr_word = (text_to_ipa_all_words(asr_text).get("joined_word", "") or "").lower() if asr_text else ""

        target_ipa = get_target_ipa(target_word)

        asr_ipa_pack = text_to_ipa_all_words(asr_text) if asr_text else {
            "tokens": [],
            "ipa_tokens": [],
            "ipa_text": "",
            "joined_word": "",
            "joined_ipa": "",
        }

        return {
            "target_word": target_word,
            "target_ipa": target_ipa,

            # compatibility
            "asr_word": asr_word or "",
            "asr_ipa": asr_ipa_pack["ipa_text"] or "",
            "asr_text": asr_text or "",

            # joined versions
            "asr_joined_word": asr_ipa_pack["joined_word"] or "",
            "asr_joined_ipa": asr_ipa_pack["joined_ipa"] or "",

            # detail
            "asr_tokens": asr_ipa_pack["tokens"],
            "asr_ipa_tokens": asr_ipa_pack["ipa_tokens"],
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)