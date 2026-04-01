import streamlit as st
import os
import psutil
import time
import glob
import pickle
import re
import numpy as np
from mlx_lm import generate
from pypdf import PdfReader
from ebooklib import epub
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
from docx import Document

# Hämta från din centrala modul
from core_loader import load_llm, load_encoder, get_available_models, BASE_PATH

# --- KONFIGURATION FÖR ARKIVET ---
ENGRAM_BASE = os.path.join(BASE_PATH, "engrams", "user_data")
INDEX_FILE = os.path.join(ENGRAM_BASE, "master_index.json")
RAW_FOLDER = os.path.join(BASE_PATH, "raw_data")

# --- HJÄLPFUNKTIONER FÖR ARKIVERING ---

def load_master_index():
    os.makedirs(ENGRAM_BASE, exist_ok=True)
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            import json
            return json.load(f)
    return {"subjects": {}, "sub_subjects": {}, "files": {}}

def save_master_index(index):
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        import json
        json.dump(index, f, indent=4, ensure_ascii=False)

def get_id(name, mapping):
    if name in mapping:
        return mapping[name]
    new_id = f"{(len(mapping) + 1):03d}"
    mapping[name] = new_id
    return new_id

def extract_text_from_upload(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if ext == ".txt":
            return uploaded_file.read().decode("utf-8", errors="ignore")
        elif ext == ".pdf":
            reader = PdfReader(uploaded_file)
            return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif ext == ".docx":
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".rtf":
            return rtf_to_text(uploaded_file.read().decode("utf-8", errors="ignore"))
    except Exception as e:
        st.error(f"Fel vid läsning av {uploaded_file.name}: {e}")
    return None

def ai_analyze_text(text_chunk, model, tokenizer):
    prompt = f"""Analysera dokumentet och svara ENDAST med JSON.
    Kategorisera i:
    1. amne (Huvudkategori, t.ex. 'Vetenskap')
    2. underamne (Nisch, t.ex. 'Kärnfysik')
    3. nyckelord (Lista med exakt 10 ord)

    TEXT: {text_chunk[:2500]}"""

    messages = [{"role": "system", "content": "Du är en bibliotekarie som svarar enbart i JSON."},
                {"role": "user", "content": prompt}]
    
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=formatted, max_tokens=300, verbose=False)
    
    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
        import json
        return json.loads(json_str)
    except:
        return {"amne": "Osorterat", "underamne": "Allmänt", "nyckelord": []}

# --- GRÄNSSNITT-INSTÄLLNINGAR ---
st.set_page_config(page_title="M4 Librarian OS", page_icon="📚", layout="wide")

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

# --- SIDEBAR ---
st.sidebar.title("⚙️ Librarian Control")
st.sidebar.metric("System RAM (Process)", f"{get_ram_usage():.2f} GB")

st.sidebar.markdown("---")

# Modellval
model_options = get_available_models()
selected_model = st.sidebar.selectbox("Välj LLM-modell", model_options) if model_options else None
model_full_path = os.path.join(BASE_PATH, "models", selected_model) if selected_model else ""

# PÅ/AV för Arkivet
st.sidebar.subheader("🔍 Sökinställningar")
use_archive = st.sidebar.toggle("Använd Arkiv (RAG)", value=True, help="PÅ: AI:n letar i dina filer. AV: AI:n svarar fritt.")
search_threshold = st.sidebar.slider("Söktröskel (Similarity)", 0.0, 0.70, 0.22, 0.01)

st.sidebar.markdown("---")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button("🚀 Ladda Modeller"):
        if selected_model:
            with st.sidebar.status("Laddar...", expanded=False) as status:
                st.session_state.model, st.session_state.tokenizer = load_llm(model_full_path)
                enc_path = os.path.join(BASE_PATH, "models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
                st.session_state.encoder = load_encoder(enc_path)
                status.update(label="Klar!", state="complete")
            st.rerun()

with col2:
    if st.sidebar.button("🗑️ Töm RAM"):
        st.cache_resource.clear()
        for k in ['model', 'tokenizer', 'encoder']:
            if k in st.session_state: del st.session_state[k]
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown('<p style="color:#FF4B4B; font-weight:bold; font-size:14px; background-color: rgba(255,75,75,0.1); padding:10px; border-radius:5px;">⚠️ VIKTIGT: Stäng terminalfönstret eller tryck Ctrl + C när du är klar.</p>', unsafe_allow_html=True)

# --- HUVUDYTA: TABS ---
tab1, tab2 = st.tabs(["📥 Arkivera & Ingest", "💬 Bibliotekarien (Chatt)"])

# --- TAB 1: ARKIVERING ---
with tab1:
    st.header("Arkivera nya dokument")
    files = st.file_uploader("Dra in filer", accept_multiple_files=True)
    
    if files and 'model' in st.session_state:
        if st.button("Starta AI-Arkivering"):
            master_index = load_master_index()
            for f in files:
                start_f = time.perf_counter()
                st.subheader(f"📄 Bearbetar: {f.name}")
                text = extract_text_from_upload(f)
                analysis = ai_analyze_text(text, st.session_state.model, st.session_state.tokenizer)
                st.write(f"✔️ AI-kategorisering: **{analysis['amne']} > {analysis['underamne']}**")
                
                chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
                vectors = st.session_state.encoder.encode(chunks)
                
                s_id = get_id(analysis['amne'], master_index['subjects'])
                sub_id = get_id(analysis['underamne'], master_index['sub_subjects'])
                target_dir = os.path.join(ENGRAM_BASE, s_id, sub_id)
                os.makedirs(target_dir, exist_ok=True)
                
                f_id = f"{(len(glob.glob(os.path.join(target_dir, '*.tq'))) + 1):03d}"
                full_uid = f"{s_id}{sub_id}{f_id}"
                
                with open(os.path.join(target_dir, f"{full_uid}.tq"), 'wb') as out:
                    pickle.dump({'vectors': vectors, 'texts': chunks, 'metadata': analysis}, out)
                
                master_index['files'][full_uid] = {"original_name": f.name, "keywords": analysis['nyckelord']}
                save_master_index(master_index)
                st.success(f"✅ Arkiverad på {time.perf_counter() - start_f:.2f}s")

# --- TAB 2: CHATT ---
with tab2:
    st.header("Bibliotekarien")
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Fråga..."):
        if 'model' in st.session_state:
            start_total = time.perf_counter()
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)

            with st.chat_message("assistant"):
                context = ""
                search_time = 0
                
                if use_archive:
                    status = st.status("🔍 Söker i arkivet...", expanded=False)
                    start_s = time.perf_counter()
                    files_tq = glob.glob(os.path.join(ENGRAM_BASE, "**/*.tq"), recursive=True)
                    
                    if files_tq:
                        q_vec = st.session_state.encoder.encode([prompt])[0]
                        matches = []
                        for fp in files_tq:
                            with open(fp, 'rb') as f_in:
                                d = pickle.load(f_in)
                                sims = np.dot(d['vectors'], q_vec) / (np.linalg.norm(d['vectors'], axis=1) * np.linalg.norm(q_vec))
                                if np.max(sims) > search_threshold:
                                    matches.append((np.max(sims), d['texts'][np.argmax(sims)]))
                        matches.sort(key=lambda x: x[0], reverse=True)
                        context = "\n---\n".join([m[1] for m in matches[:5]])
                    search_time = time.perf_counter() - start_s
                    status.update(label=f"✅ Sökning klar ({search_time:.2f}s)", state="complete")
                    
                    system_prompt = "Du är en professionell bibliotekarie. Svara på svenska baserat på arkivet. Om info saknas, svara allmänt men var ärlig."
                else:
                    # LOGIK FÖR NÄR ARKIVET ÄR AVSTÄNGT
                    system_prompt = "Du är en hjälpsam AI-assistent. Svara på svenska. Om användaren ställer frågor om 'dem' eller specifika system utan sammanhang, be dem förtydliga eller nämn att du inte har tillgång till arkivet just nu."
                
                start_gen = time.perf_counter()
                full_p = f"KONTEXT FRÅN ARKIVET:\n{context}\n\nFRÅGA: {prompt}" if context else prompt
                
                msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": full_p}]
                inp = st.session_state.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                response = generate(st.session_state.model, st.session_state.tokenizer, prompt=inp, max_tokens=1000)
                gen_time = time.perf_counter() - start_gen
                
                st.markdown(response)
                st.caption(f"⏱️ Sök: {search_time:.2f}s | AI: {gen_time:.2f}s | Totalt: {time.perf_counter()-start_total:.2f}s")
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Ladda modeller först!")