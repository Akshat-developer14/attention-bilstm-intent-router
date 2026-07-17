import os
import re
import json
import time
import numpy as np
import joblib
import spacy
import torch
import onnxruntime as ort
from spellchecker import SpellChecker
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. SETUP PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
DATA_DIR = os.path.join(BASE_DIR, "..", "data_info")

ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "intent_classifier_lstm_model.onnx")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab_map.joblib")
LABEL_MAPPING_PATH = os.path.join(DATA_DIR, "label_mapping.json")
TEST_MESSAGES_PATH = os.path.join(BASE_DIR, "test_messages.json")

# Verify paths
for path in [ONNX_MODEL_PATH, VOCAB_PATH, LABEL_MAPPING_PATH, TEST_MESSAGES_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required path does not exist: {path}")

# 2. LOAD RESOURCES
print("Loading SpaCy en_core_web_sm model...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

print("Loading spelling corrector...")
spell = SpellChecker()

print("Loading LSTM ONNX session...")
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
word_to_idx = joblib.load(VOCAB_PATH)

with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
    raw_labels = json.load(f)
    idx_to_label = {int(k): v for k, v in raw_labels.items()}

# Load DistilBERT from Hugging Face Hub (Public model Akshatdev14/distilbert_intent_model)
print("Loading DistilBERT Tokenizer and Model from HF Hub...")
hf_model_name = "akshatdev14/distilbert_intent_model"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
db_model = AutoModelForSequenceClassification.from_pretrained(hf_model_name)
db_model.eval()

# Load Test Queries
with open(TEST_MESSAGES_PATH, "r", encoding="utf-8") as f:
    test_suite = json.load(f)

# 3. DEFINE PREPROCESSING & INFERENCE FUNCTIONS

# --- LSTM Preprocessing & Prediction Flow (As Trained) ---
FLUFF_PATTERNS = [
    r"^hello\s+(customer\s+service|support|team|department)?\b",
    r"^hi\s+(customer\s+service|support|team|department)?\b",
    r"^hey\s+(customer\s+service|support|team|department)?\b",
    r"\bi\s+am\s+writing\s+(this\s+long\s+message|to\s+you)\s+because\b",
    r"\bcould\s+you\s+please\b",
    r"\bplease\s+look\s+up\b",
    r"\blook\s+up\b"
]

def strip_conversational_fluff(text: str) -> str:
    text = text.lower().strip()
    for pattern in FLUFF_PATTERNS:
        text = re.sub(pattern, "", text).strip()
    return text

def lean_autocorrect(text):
    text = text.lower().strip()
    words = re.findall(r'\b\w+\b', text)
    misspelled = spell.unknown(words)
    corrected_words = []
    for word in words:
        if word in misspelled:
            correction = spell.correction(word)
            corrected_words.append(correction if correction else word)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

def predict_lstm(raw_query, max_len=16):
    # Step A: Typo Correction (Spell Checker active)
    clean_text = lean_autocorrect(raw_query)
    
    # Step B: Conversational Fluff stripping
    trimmed_text = strip_conversational_fluff(clean_text)
    
    # Step C: Lemmatizer (spaCy active)
    doc = nlp(trimmed_text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    
    # Step D: Indexing and Padding
    encoded = [word_to_idx.get(token, 1) for token in tokens[:max_len]]
    padded = encoded + [0] * (max_len - len(encoded))
    
    # Step E: ONNX Inference
    input_array = np.array([padded], dtype=np.int64)
    onnx_outputs = ort_session.run(None, {'input': input_array})
    logits = onnx_outputs[0]
    predicted_class = int(np.argmax(logits, axis=1)[0])
    
    return idx_to_label.get(predicted_class, "unknown_intent")

# --- DistilBERT Preprocessing & Prediction Flow (As Trained) ---
def clean_text_distilbert(text: str) -> str:
    # Lowercase, replace variable placeholders with unk_placeholder, strip punctuation/extra spaces
    text = text.lower()
    text = re.sub(r'\{\{[\s\w]+\}\}', 'unk_placeholder', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_distilbert(raw_query):
    # Step A: Standard Clean
    clean_text = clean_text_distilbert(raw_query)
    
    # Step B: WordPiece Tokenization (padding/truncation to 64 tokens)
    inputs = tokenizer(
        clean_text, 
        padding="max_length", 
        truncation=True, 
        max_length=64, 
        return_tensors="pt"
    )
    
    # Step C: PyTorch Inference (CPU)
    with torch.no_grad():
        outputs = db_model(**inputs)
        logits = outputs.logits
        predicted_idx = torch.argmax(logits, dim=1).item()
        
    return idx_to_label.get(predicted_idx, "unknown_intent")

# 4. BENCHMARK RUN
print(f"\nBenchmarking both models on {len(test_suite)} queries...")

lstm_latencies = []
db_latencies = []

lstm_correct = 0
db_correct = 0

print("-" * 90)
print(f"{'Query':<40} | {'True Intent':<15} | {'LSTM (Time)':<15} | {'DistilBERT (Time)':<15}")
print("-" * 90)

for item in test_suite:
    query = item["text"]
    true_intent = item["intent"]
    
    # Benchmark LSTM
    t0 = time.perf_counter()
    lstm_pred = predict_lstm(query)
    lstm_time = (time.perf_counter() - t0) * 1000 # convert to ms
    lstm_latencies.append(lstm_time)
    if lstm_pred == true_intent:
        lstm_correct += 1
        
    # Benchmark DistilBERT
    t0 = time.perf_counter()
    db_pred = predict_distilbert(query)
    db_time = (time.perf_counter() - t0) * 1000 # convert to ms
    db_latencies.append(db_time)
    if db_pred == true_intent:
        db_correct += 1
        
    # Print query preview
    preview = query[:37] + "..." if len(query) > 40 else query
    print(f"{preview:<40} | {true_intent:<15} | {lstm_pred:<10} ({lstm_time:.1f}ms) | {db_pred:<10} ({db_time:.1f}ms)")

# 5. GENERATE FINAL REPORT
print("\n" + "=" * 50)
print("             FINAL COMPARISON REPORT")
print("=" * 50)

print(f"Total Test Queries      : {len(test_suite)}")
print("\n--- ACCURACY PERFORMANCE ---")
print(f"Custom Attention-BiLSTM  : {lstm_correct}/{len(test_suite)} ({lstm_correct/len(test_suite)*100:.2f}%)")
print(f"Fine-Tuned DistilBERT    : {db_correct}/{len(test_suite)} ({db_correct/len(test_suite)*100:.2f}%)")

print("\n--- LATENCY (INFERENCE SPEED ON CPU) ---")
print("Custom Attention-BiLSTM:")
print(f"  Average Latency       : {np.mean(lstm_latencies):.2f} ms")
print(f"  Median Latency        : {np.median(lstm_latencies):.2f} ms")
print(f"  95th Percentile Latency: {np.percentile(lstm_latencies, 95):.2f} ms")
print(f"  Total Benchmarking Time: {sum(lstm_latencies)/1000:.3f} seconds")

print("\nFine-Tuned DistilBERT:")
print(f"  Average Latency       : {np.mean(db_latencies):.2f} ms")
print(f"  Median Latency        : {np.median(db_latencies):.2f} ms")
print(f"  95th Percentile Latency: {np.percentile(db_latencies, 95):.2f} ms")
print(f"  Total Benchmarking Time: {sum(db_latencies)/1000:.3f} seconds")

print("\n" + "=" * 50)
