from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Review(BaseModel):
    review_text: str


@app.get("/")
def root():
    return {"message": "Welcome to insightly!"}


@app.post("/analyze_review")
def analyze_review(review: Review):
    review_text = review.review_text.lower()

    # import the model here


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)