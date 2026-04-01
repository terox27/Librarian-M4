# v00.00.07
import streamlit as st
import os
import psutil
import time
import glob
import pickle
import re
import numpy as np
import shutil
from mlx_lm import generate
from pypdf import PdfReader
from ebooklib import epub
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
from docx import Document

# Laddar funktioner från din centrala modul
from core_loader import load_main_system, BASE_PATH, ENGRAM_BASE, INDEX_FILE, load_master_index, save_master_index, get_id

# --- KONFIGURATION FÖR ARKIVET ---
RAW_FOLDER = os.path.join(BASE_PATH, "raw_data")
DONE_FOLDER = os.path.join(BASE_PATH, "arkiverat_original")
os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(DONE_FOLDER, exist_ok=True)

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
        st.error(f"Fel vid läsning: {e}")
    return None

def ai_analyze_text(text_chunk, model, tokenizer):
    # Categorization in English for better consistency with English sources
    prompt = f"Analyze the document and answer ONLY with JSON. Categorize in ENGLISH: amne (Main subject), underamne (Niche), and 10 keywords.\n\nTEXT: {text_chunk[:2500]}"
    messages = [{"role": "system", "content": "You are a professional librarian. Answer only in JSON format."},
                {"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=formatted, max_tokens=300, verbose=False)
    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
        import json
        return json.loads(json_str)
    except:
        return {"amne": "Unsorted", "underamne": "General", "nyckelord": []}

# --- GUI ---
st.set_page_config(page_title="Librarian OS v00.00.02", page_icon="🍏", layout="wide")

st.sidebar.title("🍏 Librarian OS v2")
st.sidebar.metric("RAM", f"{psutil.Process(os.getpid()).memory_info().rss / (1024**3):.2f} GB")

st.sidebar.markdown("---")
use_archive = st.sidebar.toggle("Använd Arkiv (RAG)", value=True)
search_threshold = st.sidebar.slider("Söktröskel", 0.0, 0.70, 0.22, 0.01)
# v00.00.05 - Slider för kontext-djup
top_k = st.sidebar.slider("Kontext-djup (Antal delar)", 1, 20, 5)

if st.sidebar.button("🚀 Starta Systemet"):
    with st.sidebar.status("Laddar modeller...") as status:
        m, t, e = load_main_system()
        st.session_state.model, st.session_state.tokenizer, st.session_state.encoder = m, t, e
        status.update(label="System ONLINE!", state="complete")
    st.rerun()

if st.sidebar.button("🗑️ Töm RAM"):
    st.cache_resource.clear()
    st.rerun()

# --- TABS ---
tab1, tab2 = st.tabs(["📥 Arkivera", "💬 Chatt"])

with tab1:
    st.subheader("Batch-arkivering (KINGSTON)")
    st.info(f"Lägg dina filer i: `{RAW_FOLDER}` och tryck på knappen nedan.")

    if st.button("🚀 Starta arkivering från raw_data"):
        if 'model' not in st.session_state:
            st.error("Starta systemet i sidebaren först!")
        else:
            raw_files = []
            for ext in ["*.pdf", "*.epub", "*.docx", "*.txt", "*.rtf"]:
                raw_files.extend(glob.glob(os.path.join(RAW_FOLDER, ext)))
            
            if not raw_files:
                st.warning("📭 Inga nya filer hittades i raw_data.")
            else:
                master_index = load_master_index()
                progress_bar = st.progress(0)
                
                for i, fp in enumerate(raw_files):
                    file_name = os.path.basename(fp)
                    st.write(f"📖 Bearbetar: {file_name}...")
                    
                    # Här använder vi din befintliga extraherings-logik (men för lokala filer)
                    with open(fp, "rb") as f_bytes:
                        # Vi simulerar ett uploaded_file-objekt för att återanvända din kod
                        from io import BytesIO
                        fake_file = BytesIO(f_bytes.read())
                        fake_file.name = file_name
                        text = extract_text_from_upload(fake_file)

                    if text:
                        analysis = ai_analyze_text(text, st.session_state.model, st.session_state.tokenizer)
                        
                        # Vektorisering
                        chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
                        vectors = st.session_state.encoder.encode(chunks)
                        
                        # Skapa ID och mappar
                        s_name = analysis.get('amne', 'Osorterat')
                        sub_name = analysis.get('underamne', 'Allmänt')
                        s_id = get_id(s_name, master_index['subjects'])
                        sub_id = get_id(sub_name, master_index['sub_subjects'])
                        
                        target_dir = os.path.join(ENGRAM_BASE, s_id, sub_id)
                        os.makedirs(target_dir, exist_ok=True)
                        
                        f_id = f"{(len(glob.glob(os.path.join(target_dir, '*.tq'))) + 1):03d}"
                        full_uid = f"{s_id}{sub_id}{f_id}"
                        rel_path = f"{s_id}/{sub_id}/{full_uid}.tq"
                        
                        # Spara engram
                        with open(os.path.join(target_dir, f"{full_uid}.tq"), 'wb') as out:
                            pickle.dump({'vectors': vectors, 'texts': chunks, 'metadata': analysis}, out)
                        
                        # UPPDATERAD INDEX-LOGIK v00.00.02
                        master_index['files'][full_uid] = {
                            "original_name": file_name,
                            "subject": s_name,
                            "sub_subject": sub_name,
                            "keywords": analysis.get('nyckelord', []),
                            "path": rel_path
                        }
                        save_master_index(master_index)
                        
                        # Efter lyckad arkivering: Flytta filen
                        shutil.move(fp, os.path.join(DONE_FOLDER, file_name))
                        st.success(f"✅ {file_name} är klar!")
                    
                    progress_bar.progress((i + 1) / len(raw_files))
                st.balloons()

    files = st.file_uploader("Ladda upp dokument", accept_multiple_files=True)
    if files and 'model' in st.session_state and st.button("Starta Arkivering"):
        master_index = load_master_index()
        for f in files:
            t0 = time.perf_counter()
            st.write(f"📖 Bearbetar: {f.name}...")
            
            text = extract_text_from_upload(f)
            analysis = ai_analyze_text(text, st.session_state.model, st.session_state.tokenizer)
            
            # Vektorisering
            chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
            vectors = st.session_state.encoder.encode(chunks)
            
            # Skapa ID och mappar
            s_name = analysis.get('amne', 'Osorterat')
            sub_name = analysis.get('underamne', 'Allmänt')
            s_id = get_id(s_name, master_index['subjects'])
            sub_id = get_id(sub_name, master_index['sub_subjects'])
            
            target_dir = os.path.join(ENGRAM_BASE, s_id, sub_id)
            os.makedirs(target_dir, exist_ok=True)
            
            f_id = f"{(len(glob.glob(os.path.join(target_dir, '*.tq'))) + 1):03d}"
            full_uid = f"{s_id}{sub_id}{f_id}"
            rel_path = f"{s_id}/{sub_id}/{full_uid}.tq"
            
            # Spara engram
            with open(os.path.join(target_dir, f"{full_uid}.tq"), 'wb') as out:
                pickle.dump({'vectors': vectors, 'texts': chunks, 'metadata': analysis}, out)
            
            # UPPDATERAD INDEX-LOGIK v00.00.02
            master_index['files'][full_uid] = {
                "original_name": f.name,
                "subject": s_name,
                "sub_subject": sub_name,
                "keywords": analysis.get('nyckelord', []),
                "path": rel_path
            }
            save_master_index(master_index)
            st.success(f"✅ {f.name} arkiverad som {full_uid} ({time.perf_counter()-t0:.2f}s)")

with tab2:
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Fråga arkivet..."):
        if 'model' in st.session_state:
            t_total = time.perf_counter()
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)

            with st.chat_message("assistant"):
                context, t_search = "", 0
                if use_archive:
                    t_s = time.perf_counter()
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
                        context = "\n---\n".join([m[1] for m in matches[:top_k]])
                    t_search = time.perf_counter() - t_s
                
                t_g = time.perf_counter()
                # v00.00.04: Respond in same language as user input
                full_p = f"KONTEXT:\n{context}\n\nFRÅGA: {prompt}" if context else prompt
                msgs = [
                    {
                        "role": "system",
                        "content": "You are a professional librarian. Always respond in the same language as the user's question, based on the provided context."
                    },
                    {"role": "user", "content": full_p}
                ]
                inp = st.session_state.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                response = generate(st.session_state.model, st.session_state.tokenizer, prompt=inp, max_tokens=1000)
                
                st.markdown(response)
                st.caption(f"⏱️ Sök: {t_search:.2f}s | AI: {time.perf_counter()-t_g:.2f}s | Totalt: {time.perf_counter()-t_total:.2f}s")
                st.session_state.messages.append({"role": "assistant", "content": response})
        else: st.error("Starta systemet först!")