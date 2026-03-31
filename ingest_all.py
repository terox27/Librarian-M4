import os
import pickle
import time
from glob import glob
from pypdf import PdfReader
from ebooklib import epub
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
from docx import Document
from sentence_transformers import SentenceTransformer

# --- KONFIGURATION (Samma som i din Librarian) ---
RAW_DATA_FOLDER = "/Volumes/KINGSTON/Librarian/raw_data"
OUTPUT_FOLDER = "/Volumes/KINGSTON/Librarian/engrams/user_data"
ENCODER_PATH = "/Volumes/KINGSTON/Librarian/models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

print("🚀 Laddar Vector Engine (MPS/Metal)...")
encoder = SentenceTransformer(ENCODER_PATH, device='mps')

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    try:
        if ext == ".txt":
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        elif ext == ".pdf":
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
        elif ext == ".epub":
            book = epub.read_epub(file_path)
            for item in book.get_items():
                if item.get_type() == 1: # Document type
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text() + "\n"
                    
        elif ext == ".rtf":
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = rtf_to_text(f.read())
                
        elif ext == ".docx":
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            
        return text
    except Exception as e:
        print(f"❌ Kunde inte läsa {file_path}: {e}")
        return None

def process_files():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    # Hitta alla filer i mappen
    files = []
    for ext in ["*.txt", "*.pdf", "*.epub", "*.rtf", "*.docx"]:
        files.extend(glob(os.path.join(RAW_DATA_FOLDER, ext)))
    
    if not files:
        print(f"📭 Inga filer hittades i {RAW_DATA_FOLDER}")
        return

    print(f"📦 Hittade {len(files)} filer att bearbeta.")

    for file_path in files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_FOLDER, file_name + ".tq")
        
        if os.path.exists(output_path):
            print(f"⏭️  Hoppar över {file_name} (redan konverterad).")
            continue

        print(f"📖 Bearbetar: {file_name}...")
        raw_text = extract_text(file_path)
        
        if not raw_text or len(raw_text.strip()) < 10:
            continue

        # Dela upp i bitar (Chunks) - viktigt för att sökningen ska fungera bra
        chunk_size = 1000
        overlap = 200
        chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size - overlap)]

        print(f"🧠 Skapar vektorer för {len(chunks)} block...")
        vectors = encoder.encode(chunks, show_progress_bar=True)

        # Spara som .tq (engram)
        with open(output_path, 'wb') as f:
            pickle.dump({'vectors': vectors, 'texts': chunks}, f)
        
        print(f"✅ Sparad som engram: {output_path}\n")

if __name__ == "__main__":
    process_files()
    print("🏁 Allt klart! Starta din mac_librarian.py för att börja ställa frågor.")