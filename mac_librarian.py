# v00.00.01
import os
import glob
import pickle
import time
import numpy as np
from mlx_lm import generate
# Vi hämtar "hjärnan" och inställningar från din centrala loader
from core_loader import load_llm, load_encoder, BASE_PATH

class MacLibrarian:
    def __init__(self):
        print("\n" + "="*60)
        print("🍏 M4 LIBRARIAN - TERMINAL MODE v00.00.01 🍏")
        print("="*60)
        
        # Sökvägar hämtas centralt
        self.MODEL_PATH = os.path.join(BASE_PATH, "models/Llama-3.1-8B-8bit")
        # Vi använder Multilingual ID för att matcha web_app.py
        self.ENCODER_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        # Använd core_loader för att väcka AI:n
        print(f"🧠 Ansluter till Llama via Core Loader...")
        self.model, self.tokenizer = load_llm(self.MODEL_PATH)
        
        print(f"🌍 Startar Multilingual Vector Engine (MPS)...")
        self.encoder = load_encoder(self.ENCODER_ID)
        
        self.all_vectors = []
        self.all_texts = []

    def load_all_engrams(self, folder_path):
        """Scannar KINGSTON efter alla .tq-filer och laddar dem"""
        # Vi letar i user_data mappen rekursivt
        search_pattern = os.path.join(folder_path, "user_data/**/*.tq")
        files = glob.glob(search_pattern, recursive=True)
        
        if not files:
            print(f"❌ Hittade inga engrams i {folder_path}/user_data")
            return False

        print(f"📚 Laddar in {len(files)} arkiverade filer/engrams...")
        start_t = time.time()
        
        for file_path in files:
            with open(file_path, "rb") as f:
                try:
                    data = pickle.load(f)
                    if "vectors" in data and "texts" in data:
                        self.all_vectors.append(data["vectors"])
                        self.all_texts.extend(data["texts"])
                except Exception as e:
                    print(f"⚠️ Kunde inte läsa {file_path}: {e}")
        
        if self.all_vectors:
            self.all_vectors = np.vstack(self.all_vectors)
            print(f"✅ Biblioteket redo! ({len(self.all_texts)} textblock på {time.time()-start_t:.2f}s)")
            return True
        return False
        
    def smart_search(self, question, top_k=10):
        """Vektor-sökning (Cosine Similarity)"""
        # Multilingual encoder förstår nu 'äpple' == 'apple'
        q_vec = self.encoder.encode([question])[0]
        
        # Beräkna likhet
        similarities = np.dot(self.all_vectors, q_vec) / (
            np.linalg.norm(self.all_vectors, axis=1) * np.linalg.norm(q_vec)
        )
        
        # Hämta de bästa träffarna
        best_indices = np.argsort(similarities)[-top_k:][::-1]
        return "\n\n---\n\n".join([self.all_texts[i] for i in best_indices])

# --- CHAT LOOP ---
if __name__ == "__main__":
    librarian = MacLibrarian()
    # Pekar på engrams-mappen på din KINGSTON
    LIBRARY_ROOT = os.path.join(BASE_PATH, "engrams")
    
    if librarian.load_all_engrams(LIBRARY_ROOT):
        use_retrieval = True
        print("\n--- 🍏 Librarian Terminal är redo! ---")
        print("Kommandon: '!v' (växla sökning), 'exit' (stäng)")

        while True:
            user_input = input("\n👤 Fråga: ").strip()

            if user_input.lower() in ["exit", "quit", "!e"]:
                print("Stänger ner. Glöm inte Ctrl+C!")
                break

            if user_input.lower() in ["!v", "!sök", "!växla"]:
                use_retrieval = not use_retrieval
                print(f"--- Sökning på KINGSTON: {'PÅ' if use_retrieval else 'AV'} ---")
                continue

            if not user_input: continue

            start_total = time.perf_counter()
            
            context = ""
            if use_retrieval:
                print("🔍 Letar i arkiven...")
                context = librarian.smart_search(user_input)
                system_msg = "Du är en professionell bibliotekarie. Svara på svenska baserat på arkivet."
                user_msg = f"KONTEXT FRÅN ARKIVET:\n{context}\n\nFRÅGA: {user_input}"
            else:
                system_msg = "Du är en hjälpsam assistent. Svara på svenska."
                user_msg = user_input

            # Chat Template för Llama 3.1
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            prompt = librarian.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Generera svar
            response = generate(
                librarian.model, 
                librarian.tokenizer, 
                prompt=prompt, 
                max_tokens=800, 
                verbose=False
            )
            
            total_time = time.perf_counter() - start_total

            print(f"\n🤖 Librarian:\n{'-'*40}\n{response}\n{'-'*40}")
            print(f"⏱️  Svarstid: {total_time:.2f}s")