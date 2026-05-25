from fastapi import FastAPI
from fastapi.responses import JSONResponse
from translate import Translator
from transformers import pipeline
from pydantic import BaseModel

class UserRequest(BaseModel):
    text: str

app = FastAPI()

async def classify_text(text):
    classifier = pipeline("sentiment-analysis", model="kertys/yelp_review_classifier")
    translator= Translator(from_lang="russian", to_lang="english")
    translation = translator.translate(text)
    return classifier(translation)

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.post("/classification")
async def classify(prompt: UserRequest):
    result = await classify_text(prompt.text)
    return JSONResponse({"Результат анализа:": result})