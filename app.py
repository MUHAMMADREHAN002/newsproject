from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
from gtts import gTTS
import uuid

app = FastAPI()

API_KEY = "your_secret_api_key_here"

# Load models
en_model = pipeline("summarization", model="facebook/bart-large-cnn")
multi_model_name = "csebuetnlp/mT5_multilingual_XLSum"
multi_tokenizer = AutoTokenizer.from_pretrained(multi_model_name)
multi_model = AutoModelForSeq2SeqLM.from_pretrained(multi_model_name)

# Authentication middleware
@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if request.url.path.startswith("/summarize"):
        key = request.headers.get("X-API-KEY")
        if key != API_KEY:
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)

# Request format
class SummarizeRequest(BaseModel):
    article_text: str
    bullet_count: int = 5

# Utilities
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def summarize_multilingual(text, lang_code):
    prefix = f"summarize {lang_code}: {text}"
    inputs = multi_tokenizer([prefix], return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = multi_model.generate(inputs["input_ids"], max_length=130, min_length=30, do_sample=False)
    return multi_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_text(text, lang_code, bullet_count):
    if lang_code == "en":
        summary = en_model(text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
    else:
        summary = summarize_multilingual(text, lang_code)

    bullet_list = [s.strip() for s in summary.split('.') if s.strip()]
    limited_summary = bullet_list[:bullet_count]
    return "\n".join(["• " + b for b in limited_summary]), ". ".join(limited_summary)

def generate_audio(summary_text, lang_code):
    try:
        filename = f"audio_{uuid.uuid4().hex}.mp3"
        tts = gTTS(summary_text, lang=lang_code)
        tts.save(filename)
        return filename
    except:
        return None

@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    try:
        text = req.article_text.strip()
        if len(text) < 20:
            raise HTTPException(status_code=400, detail="Text too short for summarization.")

        lang_code = detect_language(text)
        bullet_summary, clean_summary = summarize_text(text, lang_code, req.bullet_count)

        summary_file = f"summary_{uuid.uuid4().hex}.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(clean_summary)

        audio_file = generate_audio(clean_summary, lang_code)

        return {
            "summary": bullet_summary,
            "language": lang_code,
            "summary_file": f"/download/{summary_file}",
            "audio_file": f"/download/{audio_file}" if audio_file else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(filename)
