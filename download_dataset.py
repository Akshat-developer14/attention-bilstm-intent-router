"""
Downloading dataset from hugging face of banking77
"""
from datasets import load_dataset
import pandas as pd

# Load the banking dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
