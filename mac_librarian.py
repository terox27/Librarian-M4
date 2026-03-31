from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import time
import os
import glob

class MacLibrarian:
    def __init__(self):
        print("\n" + "="*60)
        print("🍏 M4 LIBRARIAN - TOTAL SSD MODE (KINGSTON) 🍏")
        print("="*60)
        
        # --- PATHS TO YOUR MODELS ON SSD ---
        # Vi byter från den gamla 4-bitarsmodellen till din nya, knivskarpa 8-bitars Llama!
        self.MODEL_PATH = "/Volumes/KINGSTON/Librarian/models/Llama-3.1-8B-8bit"
        self.ENCODER_PATH = "/Volumes/KINGSTON/Librarian/models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        # ------------------------------------------

        print(f"🧠 Loading Model from SSD...")
        self.model, self.tokenizer = load(self.MODEL_PATH)
        
        print("🔍 Starting Vector Engine (Apple Metal/MPS)...")
        self.encoder = SentenceTransformer(self.ENCODER_PATH, device='mps')
        
        self.all_vectors = []
        self.all_texts = []

    def load_all_engrams(self, folder_path):
        """Scans KINGSTON for all .tq files and loads them into RAM"""
        search_pattern = os.path.join(folder_path, "**/*.tq")
        files = glob.glob(search_pattern, recursive=True)
        
        if not files:
            print(f"❌ Error: No .tq files found in {folder_path}")
            return False

        print(f"📚 Loading {len(files)} books from library...")
        start_t = time.time()
        
        for file_path in files:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                self.all_vectors.append(data["vectors"])
                self.all_texts.extend(data["text_content"])
        
        self.all_vectors = np.vstack(self.all_vectors)
        print(f"✅ Library Ready! ({len(self.all_texts)} blocks in {time.time()-start_t:.2f}s)")
        return True
        
    def smart_search(self, question, top_k=15):
        """Finds the most relevant snippets in the entire library"""
        q_vec = self.encoder.encode(question)
        similarities = np.dot(self.all_vectors, q_vec) / (np.linalg.norm(self.all_vectors, axis=1) * np.linalg.norm(q_vec))
        best_indices = np.argsort(similarities)[-top_k:][::-1]
        return "\n\n---\n\n".join([self.all_texts[i] for i in best_indices])

 
# --- CHAT LOOP ---
if __name__ == "__main__":
    librarian = MacLibrarian()
    LIBRARY_PATH = "/Volumes/KINGSTON/Librarian/engrams"
    
    if librarian.load_all_engrams(LIBRARY_PATH):
        print("\n💡 Type your question (or 'exit' to quit).")
        
       # --- STARTA DIN LIBRARIAN ---
    use_retrieval = True
    
    print("\n--- Librarian är redo! ---")
    print("Kommandon: '!växla' (växla sökning), '!exit' (stäng)")

    while True:
        user_input = input("\n👤 Question: ").strip()

        # 1. Avsluta programmet
        if user_input.lower() in ["!exit", "exit", "quit", "!e"]:
            print("Biblioteket stänger. Hejdå!")
            break

        # 2. Växla sökläge
        if user_input.lower() in ["!v", "!sök", "!växla"]:
            use_retrieval = not use_retrieval
            print(f"--- Sökning på KINGSTON är nu {'PÅ' if use_retrieval else 'AV'} ---")
            continue

        start_total = time.perf_counter()
        search_time = 0
        
        if use_retrieval:
            print("🔍 Söker i dina engrams på KINGSTON...")
            start_search = time.perf_counter()
            # Använder din befintliga smart_search funktion
            context = librarian.smart_search(user_input, top_k=15)
            search_time = time.perf_counter() - start_search
            
            system_msg = "Du är en professionell bibliotekarie. Svara kort och koncist på svenska baserat ENBART på faktan nedan. Om faktan inte handlar om frågan, säg att du inte hittar info i arkivet men svara allmänt."
            user_msg = f"FAKTA FRÅN ARKIVET:\n{context}\n\nFRÅGA: {user_input}"
        else:
            print("🧠 Svarar enbart med allmän kunskap...")
            system_msg = "Du är en hjälpsam assistent. Svara kort och koncist på svenska."
            user_msg = user_input

        # 3. Bygg prompten med Llama 3.1 standardformat (Chat Template)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        prompt = librarian.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 4. Generera svar
        start_gen = time.perf_counter()
        response = generate(
            librarian.model, 
            librarian.tokenizer, 
            prompt=prompt, 
            max_tokens=1000, 
            verbose=False
        )
        gen_time = time.perf_counter() - start_gen
        total_time = time.perf_counter() - start_total

        print(f"\n🤖 Librarian response:\n{'-'*40}\n{response}\n{'-'*40}")
        print(f"⏱️  Tidsrapport: Sökning: {search_time:.2f}s | AI: {gen_time:.2f}s | Totalt: {total_time:.2f}s")