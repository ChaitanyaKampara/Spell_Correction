from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import math
from collections import defaultdict
from models import SessionLocal, User
import crud
# Import necessary functions and data
from spellFASTAPIhelper import get_corrections, probs, vocab

app = FastAPI()

# User models and schemas


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str

# Sentiment analysis models and schemas


class Tweet(BaseModel):
    text: str

# Spell checker models and schemas


class CorrectionRequest(BaseModel):
    word: str
    verbose: bool = False


# Placeholder for class probabilities, word probabilities, IDF, and number of documents
class_probs = {}
class_word_probs = {}
idf = {}
num_documents = 1  # Replace with actual number of documents
ngram_value = 2  # Example: Using bigrams

# Function to generate n-grams from a sentence


def generate_ngrams(text: str, n: int) -> List[str]:
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

# TF-IDF Calculation with n-grams


def calculate_tfidf(sentence: str, idf: Dict[str, float], num_documents: int, n: int) -> Dict[str, float]:
    ngrams = generate_ngrams(sentence, n)
    tf = defaultdict(int)
    for ngram in ngrams:
        tf[ngram] += 1
    tfidf_scores = {}
    for ngram, count in tf.items():
        tf_score = count / len(ngrams)
        idf_score = idf.get(ngram, math.log(num_documents + 1))
        tfidf_scores[ngram] = tf_score * idf_score
    return tfidf_scores

# IDF Calculation with n-grams


def calculate_idf(corpus: List[str], n: int) -> Tuple[Dict[str, float], int]:
    num_documents = len(corpus)
    df = defaultdict(int)
    for document in corpus:
        ngrams = set(generate_ngrams(document, n))
        for ngram in ngrams:
            df[ngram] += 1
    idf = {ngram: math.log(num_documents / (count + 1))
           for ngram, count in df.items()}
    return idf, num_documents

# Naive Bayes Training with n-grams


def train_naive_bayes(corpus: List[str], labels: List[str], n: int) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, float], int]:
    class_docs = defaultdict(list)
    for document, label in zip(corpus, labels):
        class_docs[label].append(document)
    class_probs = {label: len(docs) / len(corpus)
                   for label, docs in class_docs.items()}
    word_counts = {label: defaultdict(int) for label in class_docs}
    for label, docs in class_docs.items():
        for doc in docs:
            ngrams = generate_ngrams(doc, n)
            for ngram in ngrams:
                word_counts[label][ngram] += 1
    class_word_probs = {}
    for label, counts in word_counts.items():
        total_words = sum(counts.values())
        class_word_probs[label] = {
            ngram: (count / total_words) for ngram, count in counts.items()}
    idf, num_documents = calculate_idf(corpus, n)
    return class_probs, class_word_probs, idf, num_documents

# Predict Function with n-grams


def predict_naive_bayes(sentence: str, class_probs: Dict[str, float], class_word_probs: Dict[str, Dict[str, float]], idf: Dict[str, float], num_documents: int, n: int) -> str:
    tfidf_scores = calculate_tfidf(sentence, idf, num_documents, n)
    class_scores = {label: math.log(prob)
                    for label, prob in class_probs.items()}
    for label, word_probs in class_word_probs.items():
        for ngram, score in tfidf_scores.items():
            word_prob = word_probs.get(
                ngram, 1e-7 / (len(word_probs) + len(tfidf_scores)))
            class_scores[label] += math.log(word_prob * (score + 1e-7))
    predicted_class = max(class_scores, key=class_scores.get)
    return predicted_class


# Sentiment Encoding
encodingSentiment = {"positive": 2, "negative": 0, "neutral": 1}


def encoding_sentiments(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset.replace(encodingSentiment, inplace=True)
    return dataset


@app.post("/register")
def register(request: RegisterRequest, db: Session = Depends(lambda: SessionLocal())):
    user = crud.get_user_by_username(db, request.username)
    if user:
        raise HTTPException(
            status_code=400, detail="Username already registered")
    crud.create_user(db, request.username, request.password)
    return {"message": "User created successfully"}


@app.post("/login")
def login(request: LoginRequest, db: Session = Depends(lambda: SessionLocal())):
    user = crud.get_user_by_username(db, request.username)
    if not user or user.password != request.password:
        raise HTTPException(
            status_code=400, detail="Invalid username or password")
    return {"message": "Login successful"}


@app.post("/predict")
def predict(tweet: Tweet, db: Session = Depends(lambda: SessionLocal())):
    sentiment = predict_naive_bayes(
        tweet.text, class_probs, class_word_probs, idf, num_documents, ngram_value)
    return {"sentiment": sentiment}


@app.post("/corrections")
async def corrections_post(request: CorrectionRequest):
    try:
        result = get_corrections(request.word, probs, vocab, request.verbose)
        return {"corrections": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/corrections")
async def corrections_get(word: str, verbose: Optional[bool] = Query(False)):
    try:
        result = get_corrections(word, probs, vocab, verbose)
        return {"corrections": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Load your training data
train_data_path = r"C:\Users\CHAITANYA\Documents\SENTIMENTAL_ANALYSIS_NLP_PROJECT\Train__Data.csv"
train_data = pd.read_csv(train_data_path)
test_data_path = r"C:\Users\CHAITANYA\Documents\SENTIMENTAL_ANALYSIS_NLP_PROJECT\Test_Data.csv"
test_data = pd.read_csv(test_data_path)

# Encode sentiments
train_data = encoding_sentiments(train_data)
test_data = encoding_sentiments(test_data)

# Ensure all text entries are strings and fill missing values
train_data['text'] = train_data['text'].astype(str).fillna("")
test_data['text'] = test_data['text'].astype(str).fillna("")

# Assuming the columns are 'text' and 'sentiment' (adjust as necessary)
corpus = train_data['text'].tolist()
labels = train_data['sentiment'].tolist()

# Preprocess and train the model with the actual data
class_probs, class_word_probs, idf, num_documents = train_naive_bayes(
    corpus, labels, ngram_value)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
