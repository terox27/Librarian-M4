# v00.00.01
import os
import streamlit as st
from mlx_lm import load
from sentence_transformers import SentenceTransformer

# --- KONFIGURATION ---
BASE_PATH = "/Volumes/KINGSTON/Librarian"

# Standardstigar för "Starta Systemet"-knappen
MAIN_LLM_PATH = os.path.join(BASE_PATH, "models/Llama-3.1-8B-8bit")
# Denna laddas ner automatiskt om den saknas (ca 420MB)
ENCODER_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_resource
def load_llm(model_path):
    """Laddar Llama-modellen i minnet (MPS/Metal)."""
    print(f"🧠 Laddar LLM från: {model_path}")
    return load(model_path)

@st.cache_resource
def load_encoder(encoder_path):
    """Laddar SentenceTransformer (Multilingual Vector Engine)."""
    print(f"🚀 Laddar Encoder: {encoder_path}")
    return SentenceTransformer(encoder_path, device='mps')

@st.cache_resource
def load_main_system():
    """Hjälpfunktion för att väcka hela biblioteket samtidigt."""
    model, tokenizer = load_llm(MAIN_LLM_PATH)
    encoder = load_encoder(ENCODER_ID)
    return model, tokenizer, encoder

def get_available_models():
    """Hittar alla mappar i din models-katalog på KINGSTON."""
    model_dir = os.path.join(BASE_PATH, "models")
    if not os.path.exists(model_dir):
        return []
    
    models = [d for d in os.listdir(model_dir) 
              if os.path.isdir(os.path.join(model_dir, d)) and not d.startswith('.')]
    return models

# Exportera variabler för användning i andra filer
__all__ = ['load_llm', 'load_encoder', 'load_main_system', 'get_available_models', 'BASE_PATH']