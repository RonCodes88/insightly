import re
from pathlib import Path
import pickle
from typing import Any, Dict, Callable, List

import joblib
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch import nn
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
    model: str = "naive_bayes"


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
NAIVE_BAYES_MODEL_PATH = BASE_DIR / "models" / "naive_bayes_pipeline.pkl"
RNN_VOCAB_PATH = PROJECT_ROOT / "rnn_lstm" / "ecommerce_dataset" / "vocab.pkl"
RNN_LABEL_MAP_PATH = PROJECT_ROOT / "rnn_lstm" / "ecommerce_dataset" / "label_mapping.pkl"
RNN_STATE_PATH = PROJECT_ROOT / "rnn_lstm" / "ecommerce_dataset" / "baseline_rnn.pt"
SEQUENCE_LENGTH_PATH = PROJECT_ROOT / "rnn_lstm" / "ecommerce_dataset" / "sequence_length.txt"


def _ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise RuntimeError(f"Required artifact is missing: {path}")
    return path


def _load_pickle(path: Path) -> Any:
    with _ensure_exists(path).open("rb") as file:
        return pickle.load(file)


def _load_sequence_length(path: Path) -> int:
    return int(_ensure_exists(path).read_text().strip())


naive_bayes_pipeline = joblib.load(_ensure_exists(NAIVE_BAYES_MODEL_PATH))
rnn_vocab: Dict[str, Any] = _load_pickle(RNN_VOCAB_PATH)
rnn_label_mapping: Dict[str, Any] = _load_pickle(RNN_LABEL_MAP_PATH)
sequence_length = _load_sequence_length(SEQUENCE_LENGTH_PATH)


class SentimentRNN(nn.Module):
    """Baseline RNN for 3-class sentiment classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 64,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        outputs, _ = self.rnn(embedded)
        last_output = outputs[:, -1, :]
        logits = self.fc(last_output)
        return logits


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
num_classes = len(rnn_label_mapping)
rnn_model = SentimentRNN(vocab_size=len(rnn_vocab), num_classes=num_classes)
rnn_state_dict = torch.load(_ensure_exists(RNN_STATE_PATH), map_location=DEVICE)
rnn_model.load_state_dict(rnn_state_dict)
rnn_model.to(DEVICE)
rnn_model.eval()

# Create reverse mapping for label indices to names
idx_to_label = {v: k for k, v in rnn_label_mapping.items()}


def _predict_naive_bayes(text: str) -> Dict[str, Any]:
    predicted_score = int(naive_bayes_pipeline.predict([text])[0])
    probability_vector = naive_bayes_pipeline.predict_proba([text])[0].tolist()
    return {
        "model": "naive_bayes",
        "predicted_score": predicted_score,
        "probabilities": probability_vector,
        "confidence": max(probability_vector) * 100.0,
    }


CLEANING_REGEX = re.compile(r"[^a-z\s]")
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def _clean_text(text: str) -> str:
    lowered = text.lower()
    stripped = CLEANING_REGEX.sub("", lowered)
    return " ".join(stripped.split())


def _tokenize(text: str) -> List[str]:
    return text.split()


def _tokens_to_sequence(tokens: List[str]) -> List[int]:
    unk_idx = rnn_vocab.get(UNK_TOKEN, 1)
    return [rnn_vocab.get(token, unk_idx) for token in tokens]


def _pad_sequence(sequence: List[int], max_len: int, pad_value: int = 0) -> List[int]:
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + [pad_value] * (max_len - len(sequence))


def _predict_rnn(text: str) -> Dict[str, Any]:
    cleaned = _clean_text(text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Review text lacks valid tokens.")

    tokens = _tokenize(cleaned)
    sequence = _tokens_to_sequence(tokens)
    padded = _pad_sequence(sequence, sequence_length, pad_value=rnn_vocab.get(PAD_TOKEN, 0))
    input_tensor = torch.tensor([padded], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits = rnn_model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    sentiment_label = idx_to_label[pred_idx]
    
    # Map 3-class probabilities to 1-5 rating scale using weighted sum
    probs_list = probs[0].tolist()
    
    # Find indices for each sentiment (handle case-insensitive matching)
    sentiment_indices = {}
    for idx, label in idx_to_label.items():
        sentiment_indices[label.lower()] = idx
    
    neg_idx = sentiment_indices.get("negative", 0)
    neu_idx = sentiment_indices.get("neutral", 1)
    pos_idx = sentiment_indices.get("positive", 2)
    
    # Weighted rating: negative contributes to 1-2, neutral to 3, positive to 4-5
    rating_estimate = (
        probs_list[neg_idx] * 1.5 +  # negative → ~1-2
        probs_list[neu_idx] * 3.0 +   # neutral → 3
        probs_list[pos_idx] * 4.5     # positive → ~4-5
    )
    
    predicted_score = max(1, min(5, round(rating_estimate)))
    
    return {
        "model": "rnn",
        "predicted_score": predicted_score,
        "sentiment": sentiment_label,
        "confidence": confidence * 100.0,
        "probabilities": probs_list,
    }


MODEL_PREDICTORS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    "naive_bayes": _predict_naive_bayes,
    "rnn": _predict_rnn,
}


@app.get("/")
def root():
    return {"message": "Welcome to insightly!"}


@app.post("/analyze_review")
def analyze_review(review: Review):
    review_text = review.review_text.strip()
    if not review_text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    lowered_text = review_text.lower()
    selected_model = review.model.lower().strip()

    predictor = MODEL_PREDICTORS.get(selected_model)
    if predictor is None:
        supported = ", ".join(sorted(MODEL_PREDICTORS.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{review.model}'. Available options: {supported}.",
        )

    try:
        return predictor(lowered_text)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail="Failed to generate prediction.") from exc


if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)