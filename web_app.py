import streamlit as st
import os
import psutil
import time
import glob
import pickle
import numpy as np
from mlx_lm import generate
from core_loader import load_llm, load_encoder, get_available_models, BASE_PATH

# --- GRÄNSSNITT-INSTÄLLNINGAR ---
st.set_page_config(page_title="M4 Librarian OS", page_icon="📚", layout="wide")

# Funktion för att kolla RAM-användning
def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)  # GB

# --- SIDEBAR: KONTROLLPANEL ---
st.sidebar.title("⚙️ Librarian Control")

# RAM-Monitor
ram_used = get_ram_usage()
st.sidebar.metric("System RAM (Process)", f"{ram_used:.2f} GB")

st.sidebar.markdown("---")

# Modellval
model_options = get_available_models()
if model_options:
    selected_model = st.sidebar.selectbox("Välj LLM-modell", model_options)
    model_full_path = os.path.join(BASE_PATH, "models", selected_model)
else:
    st.sidebar.error("Inga modeller hittades på KINGSTON.")
    selected_model = None

# Knappar för laddning
if st.sidebar.button("🚀 Ladda Modeller"):
    if selected_model:
        with st.sidebar.status("Laddar in i M4 RAM...") as status:
            st.session_state.model, st.session_state.tokenizer = load_llm(model_full_path)
            # Standardstig till din encoder
            encoder_path = os.path.join(BASE_PATH, "models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
            st.session_state.encoder = load_encoder(encoder_path)
            status.update(label="Systemet är redo!", state="complete")
        st.rerun()

if st.sidebar.button("🗑️ Töm RAM"):
    st.cache_resource.clear()
    for key in ['model', 'tokenizer', 'encoder']:
        if key in st.session_state: del st.session_state[key]
    st.rerun()

# RÖD VARNINGSTEXT
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="color:#FF4B4B; font-weight:bold; font-size:14px; background-color: rgba(255,75,75,0.1); padding:10px; border-radius:5px;">'
    '⚠️ VIKTIGT:<br>Stäng terminalfönstret eller tryck Ctrl + C när du är klar för att frigöra RAM-minnet på din Mac.'
    '</p>', 
    unsafe_allow_html=True
)

# --- HUVUDYTA: TABS ---
tab1, tab2 = st.tabs(["📥 Arkivera & Ingest", "💬 Bibliotekarien (Chatt)"])

# --- TAB 1: ARKIVERING ---
with tab1:
    st.header("Arkivera till user_data")
    uploaded_files = st.file_uploader("Dra in filer (PDF, EPUB, TXT)", accept_multiple_files=True)
    
    if uploaded_files and 'model' in st.session_state:
        if st.button("Starta AI-Arkivering"):
            # Här kan du senare importera och anropa din run_archiver funktion
            st.warning("Arkiverings-logiken körs nu via din librarian_archiver.py logik.")
            for uploaded_file in uploaded_files:
                st.write(f"✅ Mottagit: {uploaded_file.name}")
    elif uploaded_files:
        st.error("Ladda modellen i sidebaren först!")

# --- TAB 2: CHATT (DEN NYA LOGIKEN) ---
with tab2:
    st.header("Fråga ditt arkiv")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Skriv din fråga här..."):
        if 'model' in st.session_state:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                status_box = st.status("🔍 Söker i dina engrams på KINGSTON...", expanded=False)
                
                # 1. Hitta alla .tq filer i user_data
                engram_path = os.path.join(BASE_PATH, "engrams/user_data/**/*.tq")
                files = glob.glob(engram_path, recursive=True)
                
                if not files:
                    status_box.update(label="❌ Inga arkiverade filer hittades.", state="error")
                    response = "Jag hittade inga filer i ditt arkiv (user_data). Har du kört arkiveringen?"
                else:
                    # 2. Vektorisera frågan
                    query_vec = st.session_state.encoder.encode([prompt])[0]
                    best_matches = []

                    # 3. Sök igenom filerna
                    for f_path in files:
                        with open(f_path, 'rb') as f:
                            data = pickle.load(f)
                            # Cosine Similarity
                            sims = np.dot(data['vectors'], query_vec) / (
                                np.linalg.norm(data['vectors'], axis=1) * np.linalg.norm(query_vec)
                            )
                            max_idx = np.argmax(sims)
                            if sims[max_idx] > 0.25: # Tröskel för relevans
                                best_matches.append((sims[max_idx], data['texts'][max_idx]))

                    # 4. Sortera och plocka de 5 bästa styckena
                    best_matches.sort(key=lambda x: x[0], reverse=True)
                    context = "\n---\n".join([m[1] for m in best_matches[:5]])
                    
                    status_box.update(label=f"✅ Hittade {len(best_matches)} relevanta stycken. Genererar svar...", state="complete")

                    # 5. Generera svar med Llama
                    system_prompt = "Du är en professionell bibliotekarie. Svara på svenska baserat på faktan nedan. Om informationen saknas, svara utifrån din egen kunskap men var tydlig med det."
                    full_prompt = f"KONTEXT FRÅN ARKIVET:\n{context}\n\nFRÅGA: {prompt}\n\nSVAR:"
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt}
                    ]
                    formatted_input = st.session_state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    response = generate(st.session_state.model, st.session_state.tokenizer, prompt=formatted_input, max_tokens=1000)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Ladda modellen i sidebaren innan du kan chatta.")