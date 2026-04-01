# v00.00.06
import os
import pickle
import time
import shutil
from glob import glob

# --- HÄMTA FRÅN DIN CENTRALA MODUL ---
from core_loader import load_llm, load_encoder, BASE_PATH, ENGRAM_BASE, INDEX_FILE, load_master_index, save_master_index, get_id, extract_text, ai_analyze

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

        print("🧠 AI analyserar kategorier...")
        analysis = ai_analyze(full_text, model, tokenizer)
        s_name = analysis.get('amne', 'Osorterat')
        sub_name = analysis.get('underamne', 'Allmänt')
        
        s_id = get_id(s_name, index['subjects'])
        sub_id = get_id(sub_name, index['sub_subjects'])
        
        target_dir = os.path.join(ENGRAM_BASE, s_id, sub_id)
        os.makedirs(target_dir, exist_ok=True)
        
        existing_tq = glob(os.path.join(target_dir, "*.tq"))
        f_id = f"{(len(existing_tq) + 1):03d}"
        full_uid = f"{s_id}{sub_id}{f_id}"
        rel_path = f"{s_id}/{sub_id}/{full_uid}.tq"
        
        print(f"💾 Arkiverar som {s_id}-{sub_id}-{f_id} ({s_name} > {sub_name})")
        
        chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 800)]
        vectors = encoder.encode(chunks)
        
        output_file = os.path.join(target_dir, f"{full_uid}.tq")
        with open(output_file, 'wb') as f:
            pickle.dump({'vectors': vectors, 'texts': chunks, 'metadata': analysis}, f)
        
        # FIX v00.00.02: Lägger till alla fält i indexet
        index['files'][full_uid] = {
            "original_name": file_name,
            "subject": s_name,
            "sub_subject": sub_name,
            "keywords": analysis.get('nyckelord', []),
            "path": rel_path
        }
        save_master_index(index)
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