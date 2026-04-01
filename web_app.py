# v00.00.08
import streamlit as st
import os
import psutil
import time
import glob
import pickle
import re
import numpy as np
import shutil

# Laddar funktioner från din centrala modul
from core_loader import load_main_system, BASE_PATH, ENGRAM_BASE, INDEX_FILE, load_master_index, save_master_index, get_id, extract_text, ai_analyze

# --- KONFIGURATION FÖR ARKIVET ---
RAW_FOLDER = os.path.join(BASE_PATH, "raw_data")
DONE_FOLDER = os.path.join(BASE_PATH, "arkiverat_original")
os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(DONE_FOLDER, exist_ok=True)

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
    if 'engram_cache' in st.session_state: del st.session_state.engram_cache
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
                    
                    with open(fp, "rb") as f_in:
                        text = extract_text(f_in, file_name)

                    if text:
                        analysis = ai_analyze(text, st.session_state.model, st.session_state.tokenizer)
                        
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
                        if 'engram_cache' in st.session_state: del st.session_state.engram_cache
                    
                    progress_bar.progress((i + 1) / len(raw_files))
                st.balloons()

    files = st.file_uploader("Ladda upp dokument", accept_multiple_files=True)
    if files and 'model' in st.session_state and st.button("Starta Arkivering"):
        master_index = load_master_index()
        for f in files:
            t0 = time.perf_counter()
            st.write(f"📖 Bearbetar: {f.name}...")
            
            text = extract_text(f, f.name)
            analysis = ai_analyze(text, st.session_state.model, st.session_state.tokenizer)
            
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
            if 'engram_cache' in st.session_state: del st.session_state.engram_cache

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
                    if 'engram_cache' not in st.session_state:
                        files_tq = glob.glob(os.path.join(ENGRAM_BASE, "**/*.tq"), recursive=True)
                        cache = {'vectors': [], 'texts': []}
                        for fp in files_tq:
                            with open(fp, 'rb') as f_in:
                                d = pickle.load(f_in)
                                cache['vectors'].append(d['vectors'])
                                cache['texts'].extend(d['texts'])
                        if cache['vectors']:
                            cache['vectors'] = np.vstack(cache['vectors'])
                        st.session_state.engram_cache = cache
                    
                    cache = st.session_state.engram_cache
                    if cache['texts']:
                        q_vec = st.session_state.encoder.encode([prompt])[0]
                        sims = np.dot(cache['vectors'], q_vec) / (np.linalg.norm(cache['vectors'], axis=1) * np.linalg.norm(q_vec))
                        best_idx = np.where(sims > search_threshold)[0]
                        if len(best_idx) > 0:
                            sorted_idx = best_idx[np.argsort(sims[best_idx])][::-1]
                            context = "\n---\n".join([cache['texts'][i] for i in sorted_idx[:top_k]])

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