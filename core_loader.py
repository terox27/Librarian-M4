import os
import streamlit as st
from mlx_lm import load
from sentence_transformers import SentenceTransformer

# --- KONFIGURATION ---
BASE_PATH = "/Volumes/KINGSTON/Librarian"

# Vi använder Streamlits cache så att modellerna stannar i RAM
# men bara laddas när vi faktiskt anropar dem.

@st.cache_resource
def load_llm(model_path):
    """Laddar Llama-modellen i minnet."""
    print(f"🧠 Laddar LLM från: {model_path}")
    model, tokenizer = load(model_path)
    return model, tokenizer

@st.cache_resource
def load_encoder(encoder_path):
    """Laddar SentenceTransformer (Vector Engine) i minnet."""
    print(f"🚀 Laddar Encoder från: {encoder_path}")
    return SentenceTransformer(encoder_path, device='mps')

def get_available_models():
    """Hittar alla mappar i din models-katalog på KINGSTON."""
    model_dir = os.path.join(BASE_PATH, "models")
    if not os.path.exists(model_dir):
        print(f"⚠️ Hittade inte models-mappen på {model_dir}")
        return []
    
    # Hittar mappar som inte börjar med punkt (.)
    models = [d for d in os.listdir(model_dir) 
              if os.path.isdir(os.path.join(model_dir, d)) and not d.startswith('.')]
    return models

# Exportera BASE_PATH så att andra filer kan använda den
__all__ = ['load_llm', 'load_encoder', 'get_available_models', 'BASE_PATH']