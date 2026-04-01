# v00.00.06
import os
import glob
import pickle
import time
import numpy as np
from mlx_lm import generate
from core_loader import load_llm, load_encoder, BASE_PATH, load_engram_cache, perform_search

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
        self.cache = None

    def load_all_engrams(self, folder_path):
        """Scannar KINGSTON efter alla .tq-filer och laddar dem"""
        start_t = time.time()
        self.cache = load_engram_cache()
        if self.cache['texts']:
            print(f"✅ Biblioteket redo! ({len(self.cache['texts'])} textblock på {time.time()-start_t:.2f}s)")
            return True
        return False
        
    def smart_search(self, question, top_k=10):
        """Vektor-sökning (Cosine Similarity)"""
        return perform_search(question, self.encoder, self.cache, top_k=top_k)

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
                # v00.00.04: svara på samma språk som användarens fråga
                system_msg = "You are a professional librarian. Always respond in the same language as the user's question."
                user_msg = f"KONTEXT FRÅN ARKIVET:\n{context}\n\nFRÅGA: {user_input}"
            else:
                system_msg = "You are a helpful assistant. Respond in the same language as the user."
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