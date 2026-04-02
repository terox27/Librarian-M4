# v00.00.06
import os
import json
import re
import pickle
import numpy as np
import streamlit as st
import warnings
import psutil
from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
from docx import Document
import glob

# --- KONFIGURATION ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ENGRAM_BASE = os.path.join(BASE_PATH, "engrams", "user_data")
INDEX_FILE = os.path.join(ENGRAM_BASE, "master_index.json")

# Standardstigar för "Starta Systemet"-knappen
MAIN_LLM_PATH = os.path.join(BASE_PATH, "models/Llama-3.1-8B-8bit")
# Denna laddas ner automatiskt om den saknas (ca 420MB)
ENCODER_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Ignorera ebooklib-varningar för en renare konsol
warnings.filterwarnings('ignore', category=UserWarning)

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

def estimate_model_ram(model_name):
    """Uppskattar RAM-krav baserat på modellnamn (t.ex. 8B-4bit)."""
    name = model_name.lower()
    params = 8.0 # Standard om vi inte hittar annat
    
    # Försök hitta miljarder parametrar (B)
    p_match = re.search(r'(\d+(\.\d+)?)b', name)
    if p_match:
        params = float(p_match.group(1))
        
    # Bestäm bits per parameter
    if "4bit" in name or "q4" in name:
        gb_per_b = 0.7  # 4-bit tar ca 0.7GB per miljard parametrar (inkl overhead)
    elif "8bit" in name or "q8" in name:
        gb_per_b = 1.1
    elif "fp16" in name:
        gb_per_b = 2.1
    else:
        gb_per_b = 0.8  # Rimlig gissning för de flesta MLX-modeller (ofta q4/q8)
        
    return params * gb_per_b

def load_main_system(model_path):
    """Väcker AI:n med den valda modellen."""
    model, tokenizer = load_llm(model_path)
    encoder = load_encoder(ENCODER_ID)
    return model, tokenizer, encoder

def get_available_models():
    """Hittar modeller och räknar ut deras RAM-behov."""
    model_dir = os.path.join(BASE_PATH, "models")
    if not os.path.exists(model_dir):
        return []
    
    found_models = []
    for d in os.listdir(model_dir):
        path = os.path.join(model_dir, d)
        # Hoppa över dolda filer och mappen 'small' för att bara se huvudmodeller
        if os.path.isdir(path) and not d.startswith('.') and d.lower() != "small":
            found_models.append({
                "name": d,
                "path": path,
                "ram_estimate": estimate_model_ram(d)
            })
    return found_models

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
                if item.get_type() == ITEM_DOCUMENT:
                    t += BeautifulSoup(item.get_content(), 'html.parser').get_text() + "\n"
            return t
        elif ext == ".rtf":
            content = file_source.read() if hasattr(file_source, 'read') else open(file_source, 'r', encoding='utf-8', errors='ignore').read()
            text = content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else content
            return rtf_to_text(text)
    except Exception as e:
        print(f"❌ Fel vid extrahering från {filename}: {e}")
    return None

def ai_analyze(text_chunk, model, tokenizer, retries=2):
    """Kategoriserar dokumentet med hjälp av LLM med retry-logik och striktare validering."""
    prompt = (
        "Analyze the following text and provide a classification in JSON format ONLY. "
        "Keys: 'amne' (Main topic), 'underamne' (Specific sub-topic), 'nyckelord' (List of 10 keywords). "
        "Use English for values.\n\n"
        f"TEXT: {text_chunk[:2500]}"
    )
    messages = [
        {"role": "system", "content": "You are a professional librarian. Respond ONLY with a valid JSON object. No preamble or explanation."},
        {"role": "user", "content": prompt}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    for attempt in range(retries + 1):
        try:
            response = generate(model, tokenizer, prompt=formatted, max_tokens=300, verbose=False)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                # Validera att obligatoriska fält finns
                if all(k in data for k in ["amne", "underamne", "nyckelord"]):
                    return data
            
            print(f"⚠️ Försök {attempt + 1}: Ogiltig JSON-struktur från AI. Försöker igen...")
        except Exception as e:
            print(f"❌ Försök {attempt + 1}: Fel vid parsing av AI-svar: {e}")
            
    return {"amne": "Osorterat", "underamne": "Allmänt", "nyckelord": []}

def load_engram_cache():
    """Laddar alla engram-vektorer till minnet för snabb sökning."""
    files_tq = glob.glob(os.path.join(ENGRAM_BASE, "**/*.tq"), recursive=True)
    cache = {'vectors': [], 'texts': []}
    for fp in files_tq:
        try:
            with open(fp, 'rb') as f_in:
                d = pickle.load(f_in)
                cache['vectors'].append(d['vectors'])
                cache['texts'].extend(d['texts'])
        except Exception as e:
            print(f"⚠️ Kunde inte ladda {fp}: {e}")
    if cache['vectors']:
        cache['vectors'] = np.vstack(cache['vectors'])
    return cache

def perform_search(query, encoder, cache, top_k=5, threshold=0.22):
    """Utför vektorsökning (Cosine Similarity) i det cachade biblioteket."""
    if not cache or not cache.get('texts') or len(cache['texts']) == 0:
        return ""
    
    q_vec = encoder.encode([query])[0]
    sims = np.dot(cache['vectors'], q_vec) / (np.linalg.norm(cache['vectors'], axis=1) * np.linalg.norm(q_vec))
    
    best_idx = np.where(sims > threshold)[0]
    if len(best_idx) == 0: return ""
    
    sorted_idx = best_idx[np.argsort(sims[best_idx])][::-1]
    return "\n---\n".join([cache['texts'][i] for i in sorted_idx[:top_k]])

def process_and_archive(text, filename, model, tokenizer, encoder, index):
    """Gemensam logik för att analysera, vektorisera och spara ett dokument."""
    analysis = ai_analyze(text, model, tokenizer)
    
    # Skapa ID:n baserat på AI-analysen
    s_name = analysis.get('amne', 'Osorterat')
    sub_name = analysis.get('underamne', 'Allmänt')
    s_id = get_id(s_name, index['subjects'])
    sub_id = get_id(sub_name, index['sub_subjects'])
    
    target_dir = os.path.join(ENGRAM_BASE, s_id, sub_id)
    os.makedirs(target_dir, exist_ok=True)
    
    # Generera unikt fil-ID (UID)
    existing_files = glob.glob(os.path.join(target_dir, "*.tq"))
    if not existing_files:
        f_id = "001"
    else:
        # Hämta de sista 3 siffrorna i filnamnet för att hitta högsta ID
        ids = [int(os.path.basename(f)[:9][-3:]) for f in existing_files if os.path.basename(f)[:9][-3:].isdigit()]
        f_id = f"{(max(ids) + 1 if ids else len(existing_files) + 1):03d}"

    full_uid = f"{s_id}{sub_id}{f_id}"
    rel_path = f"{s_id}/{sub_id}/{full_uid}.tq"
    
    # Chunking och Vektorisering
    # Förbättrad chunking: Försök dela vid stycken för bättre kontext
    raw_chunks = re.split(r'\n\s*\n', text)
    chunks = []
    for chunk in raw_chunks:
        if len(chunk) > 1000: # Om stycket är för långt, dela det ändå
            chunks.extend([chunk[i:i+1000] for i in range(0, len(chunk), 800)])
        elif len(chunk) > 50: # Skippa för korta fragment
            chunks.append(chunk)
            
    vectors = encoder.encode(chunks)
    
    # Spara engram (vektorer + text)
    output_path = os.path.join(target_dir, f"{full_uid}.tq")
    with open(output_path, 'wb') as f:
        pickle.dump({'vectors': vectors, 'texts': chunks, 'metadata': analysis}, f)
    
    # Uppdatera master-index
    index['files'][full_uid] = {
        "original_name": filename,
        "subject": s_name,
        "sub_subject": sub_name,
        "keywords": analysis.get('nyckelord', []),
        "path": rel_path
    }
    save_master_index(index)
    
    return full_uid, s_name, sub_name

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
    'load_llm', 'load_encoder', 'load_main_system', 'get_available_models',
    'BASE_PATH', 'ENGRAM_BASE', 'INDEX_FILE',
    'load_master_index', 'save_master_index', 'get_id',
    'extract_text', 'ai_analyze', 'process_and_archive',
    'load_engram_cache', 'perform_search'
]