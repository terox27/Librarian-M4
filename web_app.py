# v00.00.11
import streamlit as st
import os
import psutil
import time
import glob
import pickle
import re
import numpy as np
import json
import shutil
from mlx_lm import generate, stream_generate

# Laddar funktioner från din centrala modul
from core_loader import load_main_system, BASE_PATH, ENGRAM_BASE, INDEX_FILE, load_master_index, save_master_index, get_id, extract_text, ai_analyze, process_and_archive, load_engram_cache, perform_search

# --- KONFIGURATION FÖR ARKIVET ---
RAW_FOLDER = os.path.join(BASE_PATH, "raw_data")
DONE_FOLDER = os.path.join(BASE_PATH, "arkiverat_original")
CONVERSATION_FILE = os.path.join(BASE_PATH, "conversation_history.json")
os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(DONE_FOLDER, exist_ok=True)

# --- HISTORIKHANTERING ---
def load_history():
    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(messages):
    with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)

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
                        full_uid, s_n, sub_n = process_and_archive(
                            text, file_name, st.session_state.model, 
                            st.session_state.tokenizer, st.session_state.encoder, master_index
                        )
                        
                        # Efter lyckad arkivering: Flytta filen
                        shutil.move(fp, os.path.join(DONE_FOLDER, file_name))
                        st.success(f"✅ {file_name} arkiverad som {full_uid} ({s_n})")
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
            if text:
                full_uid, s_n, sub_n = process_and_archive(
                    text, f.name, st.session_state.model, 
                    st.session_state.tokenizer, st.session_state.encoder, master_index
                )
                st.success(f"✅ {f.name} arkiverad som {full_uid} ({time.perf_counter()-t0:.2f}s)")
            if 'engram_cache' in st.session_state: del st.session_state.engram_cache

with tab2:
    if "messages" not in st.session_state: 
        st.session_state.messages = load_history()
    for msg in st.session_state.messages: st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Fråga arkivet..."):
        if 'model' in st.session_state:
            t_total = time.perf_counter()
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)
            save_history(st.session_state.messages)

            with st.chat_message("assistant"):
                context, t_search = "", 0
                if use_archive:
                    t_s = time.perf_counter()
                    if 'engram_cache' not in st.session_state:
                        st.session_state.engram_cache = load_engram_cache()

                    context = perform_search(
                        prompt, st.session_state.encoder, 
                        st.session_state.engram_cache, 
                        top_k=top_k, threshold=search_threshold
                    )
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

                # Streaming av svar för en mer följsam upplevelse
                placeholder = st.empty()
                full_response = ""
                for chunk in stream_generate(st.session_state.model, st.session_state.tokenizer, prompt=inp, max_tokens=1000):
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                st.caption(f"⏱️ Sök: {t_search:.2f}s | AI: {time.perf_counter()-t_g:.2f}s | Totalt: {time.perf_counter()-t_total:.2f}s")
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_history(st.session_state.messages)
        else: st.error("Starta systemet först!")