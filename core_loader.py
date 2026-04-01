# v00.00.03
import os
import json
import re
import streamlit as st
from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from ebooklib import epub
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
from docx import Document

# --- KONFIGURATION ---
BASE_PATH = "/Volumes/KINGSTON/Librarian"
ENGRAM_BASE = os.path.join(BASE_PATH, "engrams", "user_data")
INDEX_FILE = os.path.join(ENGRAM_BASE, "master_index.json")

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

# --- TEXTEXTRAKTION OCH ANALYS ---

def extract_text(file_source, filename):
    """Extraherar text från fil-sökväg eller fil-liknande objekt (stream)."""
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == ".txt":
            content = file_source.read() if hasattr(file_source, 'read') else open(file_source, 'r', encoding='utf-8', errors='ignore').read()
            return content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else content
        elif ext == ".pdf":
            reader = PdfReader(file_source)
            return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif ext == ".docx":
            doc = Document(file_source)
            return "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".epub":
            book = epub.read_epub(file_source)
            t = ""
            for item in book.get_items():
                if item.get_type() == 1:
                    t += BeautifulSoup(item.get_content(), 'html.parser').get_text() + "\n"
            return t
        elif ext == ".rtf":
            content = file_source.read() if hasattr(file_source, 'read') else open(file_source, 'r', encoding='utf-8', errors='ignore').read()
            text = content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else content
            return rtf_to_text(text)
    except Exception as e:
        print(f"❌ Fel vid extrahering från {filename}: {e}")
    return None

def ai_analyze(text_chunk, model, tokenizer):
    """Kategoriserar dokumentet med hjälp av LLM."""
    prompt = f"Analyze the document and answer ONLY with JSON. Categorize in ENGLISH: amne (Main subject), underamne (Niche), and 10 keywords.\n\nTEXT: {text_chunk[:2500]}"
    messages = [{"role": "system", "content": "You are a professional librarian. Answer only in JSON format."},
                {"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=formatted, max_tokens=300, verbose=False)
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return {"amne": "Osorterat", "underamne": "Allmänt", "nyckelord": []}

# --- INDEX-HANTERING ---

def load_master_index():
    os.makedirs(ENGRAM_BASE, exist_ok=True)
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"subjects": {}, "sub_subjects": {}, "files": {}}

def save_master_index(index):
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=4, ensure_ascii=False)

def get_id(name, mapping):
    if name in mapping:
        return mapping[name]
    new_id = f"{(len(mapping) + 1):03d}"
    mapping[name] = new_id
    return new_id

# Exportera variabler för användning i andra filer
__all__ = [
    'load_llm', 'load_encoder', 'load_main_system', 'get_available_models', 'BASE_PATH', 
    'ENGRAM_BASE', 'INDEX_FILE', 'load_master_index', 'save_master_index', 'get_id', 'extract_text', 'ai_analyze'
]