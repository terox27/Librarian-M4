import os
import streamlit as st
from mlx_lm import load
from sentence_transformers import SentenceTransformer

# Standardstigar på din KINGSTON
BASE_PATH = "/Volumes/KINGSTON/Librarian"

@st.cache_resource
def load_llm(model_path):
    print(f"🧠 Laddar LLM: {model_path}...")
    model, tokenizer = load(model_path)
    return model, tokenizer

@st.cache_resource
def load_encoder(encoder_path):
    print(f"🚀 Laddar Encoder: {encoder_path}...")
    return SentenceTransformer(encoder_path, device='mps')

def get_available_models():
    # Hittar alla mappar i din models-mapp
    model_dir = os.path.join(BASE_PATH, "models")
    if not os.path.exists(model_dir):
        return []
    return [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]