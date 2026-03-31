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

        print(f"🧠 Loading Qwen 3.5 9B from SSD...")
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

    def ask(self, question):
        """Retrieves facts and generates an English response instantly"""
        context = self.smart_search(question)
        
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a factual assistant. "
                    "Respond in English. Be concise. Go directly to the answer. "
                    "DO NOT include any 'Thinking Process' or internal reasoning. "
                    "If the information is missing, tell me what you know.'"
                )
            },
            {"role": "user", "content": f"FACTS FROM SSD:\n{context}\n\nQUESTION:\n{question}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        print("\n🤖 Librarian response:")
        print("-" * 40)
        
        # We keep only the core arguments to avoid MLX compatibility errors
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=1500,
            verbose=False
        )
        print(response)
        print("-" * 40)

# --- CHAT LOOP ---
if __name__ == "__main__":
    librarian = MacLibrarian()
    LIBRARY_PATH = "/Volumes/KINGSTON/Librarian/engrams"
    
    if librarian.load_all_engrams(LIBRARY_PATH):
        print("\n💡 Type your question (or 'exit' to quit).")
        
       # --- STARTA DIN LIBRARIAN ---
    use_retrieval = True
    
    print("\n--- Librarian är redo! ---")
    print("Kommandon: '!search' (växla sökning), '!exit' (stäng)")

    while True:
        user_input = input("\n👤 Question: ").strip()

        if user_input.lower() in ["!exit", "exit", "quit"]:
            break

        if user_input.lower() == "!search":
            use_retrieval = not use_retrieval
            print(f"--- Sökning på KINGSTON är nu {'PÅ' if use_retrieval else 'AV'} ---")
            continue

        # --- STARTA TIDTAGNING ---
        start_total = time.perf_counter()
        search_time = 0

        if use_retrieval:
            print("🔍 Söker i dina engrams på KINGSTON...")
            start_search = time.perf_counter()
            
            # Anropar din smart_search funktion
            context = librarian.smart_search(user_input, top_k=15)
            
            search_time = time.perf_counter() - start_search
            
            prompt = (
                "Du är en hjälpsam bibliotekarie. Använd följande fakta för att svara på frågan.\n\n"
                f"Fakta från KINGSTON: {context}\n\n"
                f"Fråga: {user_input}\n\nSvar:"
            )
        else:
            print("🧠 Svarar enbart med allmän kunskap...")
            prompt = f"Fråga: {user_input}\n\nSvar:"

        # --- GENERERA SVAR ---
        start_gen = time.perf_counter()
        
        response = generate(librarian.model, librarian.tokenizer, prompt=prompt, verbose=False)
        
        gen_time = time.perf_counter() - start_gen
        total_time = time.perf_counter() - start_total

        # --- SKRIV UT RESULTAT OCH TIDER ---
        print(f"\n🤖 Librarian response:\n{'-'*40}\n{response}\n{'-'*40}")
        
        print(f"⏱️  Tidsrapport:")
        if use_retrieval:
            print(f"   • Sökning på SSD:   {search_time:.2f} sekunder")
        print(f"   • AI-generering:    {gen_time:.2f} sekunder")
        print(f"   • TOTAL TID:        {total_time:.2f} sekunder")