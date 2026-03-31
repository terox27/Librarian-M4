import os
import json
import pickle
import time
import re
from glob import glob
from pypdf import PdfReader
from ebooklib import epub
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
from docx import Document
from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate  # Förutsätter att du kör MLX

# --- KONFIGURATION ---
BASE_PATH = "/Volumes/KINGSTON/Librarian"
RAW_FOLDER = os.path.join(BASE_PATH, "raw_data")
# Här låser vi allt till user_data
ENGRAM_BASE = os.path.join(BASE_PATH, "engrams", "user_data")
INDEX_FILE = os.path.join(ENGRAM_BASE, "master_index.json")

# Sökväg till din SentenceTransformer
ENCODER_PATH = os.path.join(BASE_PATH, "models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")

print("🚀 Startar Arkivarien och laddar Vector Engine...")
encoder = SentenceTransformer(ENCODER_PATH, device='mps')

# --- FUNKTIONER FÖR INDEXERING ---

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

# --- FUNKTIONER FÖR TEXTEXTRAKTION ---

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".txt":
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: return f.read()
        elif ext == ".pdf":
            return "\n".join([p.extract_text() for p in PdfReader(file_path).pages if p.extract_text()])
        elif ext == ".docx":
            return "\n".join([para.text for para in Document(file_path).paragraphs])
        elif ext == ".epub":
            book = epub.read_epub(file_path)
            t = ""
            for item in book.get_items():
                if item.get_type() == 1: 
                    t += BeautifulSoup(item.get_content(), 'html.parser').get_text() + "\n"
            return t
        elif ext == ".rtf":
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: return rtf_to_text(f.read())
    except Exception as e:
        print(f"❌ Fel vid läsning av {file_path}: {e}")
    return None

# --- AI-ANALYS ---

def ai_analyze(text_chunk, model, tokenizer):
    prompt = f"""Analysera dokumentet och svara ENDAST med JSON.
    Kategorisera i:
    1. amne (Huvudkategori, t.ex. 'Vetenskap')
    2. underamne (Nisch, t.ex. 'Kärnfysik')
    3. nyckelord (Lista med exakt 10 ord)

    TEXT: {text_chunk[:2500]}"""

    messages = [
        {"role": "system", "content": "Du är en bibliotekarie som svarar enbart i JSON."},
        {"role": "user", "content": prompt}
    ]
    
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=formatted, max_tokens=300, verbose=False)
    
    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
        return json.loads(json_str)
    except:
        return {"amne": "Osorterat", "underamne": "Allmänt", "nyckelord": []}

# --- HUVUDPROCESS ---

def run_archiver(model, tokenizer):
    index = load_master_index()
    raw_files = []
    for ext in ["*.pdf", "*.epub", "*.docx", "*.txt", "*.rtf"]:
        raw_files.extend(glob(os.path.join(RAW_FOLDER, ext)))

    if not raw_files:
        print(f"📭 Inga nya filer hittades i {RAW_FOLDER}")
        return

    for file_path in raw_files:
        file_name = os.path.basename(file_path)
        print(f"\n📖 Bearbetar: {file_name}")
        
        full_text = extract_text(file_path)
        if not full_text: continue

        print("🧠 AI analyserar kategorier...")
        analysis = ai_analyze(full_text, model, tokenizer)
        s_name, sub_name = analysis.get('amne', 'Osorterat'), analysis.get('underamne', 'Allmänt')
        
        s_id = get_id(s_name, index['subjects'])
        sub_id = get_id(sub_name, index['sub_subjects'])
        
        # FIX: Här använder vi ENGRAM_BASE som vi definierade högst upp
        target_dir = os.path.join(ENGRAM_BASE, s_id, sub_id)
        os.makedirs(target_dir, exist_ok=True)
        
        existing_tq = glob(os.path.join(target_dir, "*.tq"))
        f_id = f"{(len(existing_tq) + 1):03d}"
        full_uid = f"{s_id}{sub_id}{f_id}"
        
        print(f"💾 Arkiverar som {s_id}-{sub_id}-{f_id} ({s_name} > {sub_name})")
        
        chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 800)]
        vectors = encoder.encode(chunks)
        
        output_file = os.path.join(target_dir, f"{full_uid}.tq")
        with open(output_file, 'wb') as f:
            pickle.dump({'vectors': vectors, 'texts': chunks, 'metadata': analysis}, f)
        
        index['files'][full_uid] = {
            "original_name": file_name,
            "subject": s_name,
            "sub_subject": sub_name,
            "keywords": analysis.get('nyckelord', []),
            "path": f"{s_id}/{sub_id}/{full_uid}.tq"
        }
        save_master_index(index)
        print(f"✅ Klart! ID: {full_uid}")

if __name__ == "__main__":
    # Ändra sökvägen till där din Llama 3.1-modell ligger
    MODEL_PATH = "/Volumes/KINGSTON/Librarian/models/Llama-3.1-8B-Lexi-Uncensored-V2-4bit"
    
    print(f"📂 Laddar modell från {MODEL_PATH}...")
    model, tokenizer = load(MODEL_PATH)
    
    run_archiver(model, tokenizer)