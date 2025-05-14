import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import MBartForConditionalGeneration, MBartTokenizer
import torch
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Настройка CORS для работы с ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели и токенизатора
logger.info("Загрузка модели и токенизатора...")
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBartTokenizer.from_pretrained(model_name)

if torch.cuda.is_available():
    model = model.cuda()
    logger.info("Модель загружена на GPU")
else:
    logger.info("Модель загружена на CPU")

# Маппинг языковых кодов
MBART_LANG_MAP = {
    "ru": "ru_RU",
    "en": "en_XX"
}

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.get("/langs")
async def get_languages():
    return list(MBART_LANG_MAP.keys())

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        logger.info(f"Получен запрос на перевод: {len(request.text)} символов")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Текст не может быть пустым")

        source_lang = MBART_LANG_MAP.get(request.source_lang)
        target_lang = MBART_LANG_MAP.get(request.target_lang)

        if not source_lang or not target_lang:
            raise HTTPException(status_code=400, detail="Неподдерживаемый язык")

        # Токенизация с обработкой длинных текстов
        tokenizer.src_lang = source_lang
        encoded = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=1024)
        
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Генерация перевода с оптимизированными параметрами
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
            max_length=800,
            num_beams=8,
            length_penalty=2.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        logger.info(f"Перевод выполнен успешно: {len(translation)} символов")
        
        return {"translation": translation}

    except Exception as e:
        logger.error(f"Ошибка при переводе: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Монтирование статических файлов
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 