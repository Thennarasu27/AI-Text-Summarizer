
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services.gemini_service import generate_summary

app = FastAPI(
    title="AI Text Summarizer",
    description="An AI-powered text summarization API",
    version="1.0.0"
)


class SummaryRequest(BaseModel):
    text: str = Field(
        min_length=50,
        max_length=20000,
        description="Text that needs to be summarized"
    )


@app.get("/")
def home():
    return {
        "message": "Welcome to AI Text Summarizer API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/summarize")
def summarize_text(request: SummaryRequest):
    try:
        summary = generate_summary(request.text)

        return {
            "original_text": request.text,
            "summary": summary
        }

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Gemini API request failed. Check your API key, model access and connection."
        )
