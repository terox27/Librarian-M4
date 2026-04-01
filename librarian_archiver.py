# v00.00.07
import os
import time
import shutil
from glob import glob

# --- HÄMTA FRÅN DIN CENTRALA MODUL ---
from core_loader import load_llm, load_encoder, BASE_PATH, ENGRAM_BASE, INDEX_FILE, load_master_index, save_master_index, get_id, extract_text, ai_analyze, process_and_archive

# --- KONFIGURATION (BASERAT PÅ BASE_PATH) ---
RAW_FOLDER = os.path.join(BASE_PATH, "raw_data")
DONE_FOLDER = os.path.join(BASE_PATH, "arkiverat_original")

# --- HUVUDPROCESS ---

def run_archiver(model, tokenizer, encoder):
    index = load_master_index()
    DONE_FOLDER = os.path.join(BASE_PATH, "arkiverat_original")
    os.makedirs(DONE_FOLDER, exist_ok=True)
    raw_files = []
    for ext in ["*.pdf", "*.epub", "*.docx", "*.txt", "*.rtf"]:
        raw_files.extend(glob(os.path.join(RAW_FOLDER, ext)))

    if not raw_files:
        print(f"📭 Inga nya filer hittades i {RAW_FOLDER}")
        return

    for file_path in raw_files:
        t_start = time.perf_counter()
        file_name = os.path.basename(file_path)
        print(f"\n📖 Bearbetar: {file_name}")
        
        full_text = extract_text(file_path, file_name)
        if not full_text: continue

        print("🧠 Arkiverar...")
        full_uid, s_n, sub_n = process_and_archive(full_text, file_name, model, tokenizer, encoder, index)

        # Flytta filen till 'done'-mappen
        shutil.move(file_path, os.path.join(DONE_FOLDER, file_name))
        print(f"📦 Originalet flyttat till: {DONE_FOLDER}")
        t_total = time.perf_counter() - t_start
        print(f"✅ Klart! ID: {full_uid} (Tid: {t_total:.2f}s)")

    # Fråga om städning
    if os.listdir(DONE_FOLDER):
        clean_up = input(f"\n🧹 Arkiveringen är klar. Vill du tömma mappen '{DONE_FOLDER}'? (j/n): ").strip().lower()
        if clean_up == 'j':
            for f in glob(os.path.join(DONE_FOLDER, "*")):
                os.remove(f)
            print("✨ Mappen är nu tömd!")

# --- STARTA ---

if __name__ == "__main__":
    MODEL_PATH = os.path.join(BASE_PATH, "models/Llama-3.1-8B-8bit")
    ENCODER_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    print("🚀 Startar Arkivarien v00.00.04 via Core Loader...")
    
    model, tokenizer = load_llm(MODEL_PATH)
    encoder = load_encoder(ENCODER_ID)
    
    run_archiver(model, tokenizer, encoder)