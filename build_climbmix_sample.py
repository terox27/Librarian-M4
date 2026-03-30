from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import time
import os

# --- Inställningar ---
# Byt ut namnet till vad din Kingston XS1000 faktiskt heter i Finder
OUTPUT_DIR = "/Volumes/KINGSTON/Librarian/engrams/climbmix"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_climbmix_samples():
    print("🍏 Startar M4-optimerad skörd från ClimbMix-400B...")
    
    # 1. Streama in data direkt från Karpathys blandade version av datasetet
    print("🌐 Ansluter till karpathy/climbmix-400b-shuffle...")
    dataset = load_dataset("karpathy/climbmix-400b-shuffle", split="train", streaming=True)
    
    # Vi plockar ut 40 000 texter och delar upp dem i 4 filer (10 000 i varje)
    batch_size = 10000
    num_files = 4
    
    # 2. Starta M4:ans Neural Engine via MPS
    print("🧠 Laddar Apple Metal (MPS) Vektormotor...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')
    
    text_iterator = iter(dataset)
    
    for file_num in range(1, num_files + 1):
        print(f"\n📦 Bygger ClimbMix Del {file_num}/{num_files}...")
        start_time = time.time()
        
        # Samla in textblock
        texts = []
        for _ in range(batch_size):
            try:
                row = next(text_iterator)
                # Vi filtrerar bort extremt korta texter för att få bra fakta
                if len(row['text']) > 150: 
                    texts.append(row['text'])
            except StopIteration:
                break
                
        if not texts:
            break
            
        # Generera embeddings med M4-chippet
        print(f"   Vektoriserar {len(texts)} block...")
        embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
        
        # Spara som .tq fil på Kingston-disken
        file_path = f"{OUTPUT_DIR}/climbmix_sample_del{file_num}.tq"
        engram_payload = {
            "metadata": {"source": "ClimbMix-400B-Shuffle", "type": "M4-MLX-Sample"},
            "vectors": np.array(embeddings, dtype=np.float16),
            "text_content": texts
        }
        
        with open(file_path, "wb") as f:
            pickle.dump(engram_payload, f)
            
        print(f"   ✅ Sparad: {file_path} (Tid: {time.time() - start_time:.1f} sekunder)")

if __name__ == "__main__":
    build_climbmix_samples()
