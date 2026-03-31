import streamlit as st
import os
import psutil
import time
from core_loader import load_llm, load_encoder, get_available_models, BASE_PATH

# --- GRÄNSSNITT-INSTÄLLNINGAR ---
st.set_page_config(page_title="M4 Librarian OS", page_icon="📚", layout="wide")

# Funktion för att kolla RAM-användning på Macen
def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)  # Returnerar GB

# --- SIDEBAR: KONTROLLPANEL ---
st.sidebar.title("⚙️ Librarian Control")

# RAM-Monitor
ram_used = get_ram_usage()
st.sidebar.metric("System RAM (Process)", f"{ram_used:.2f} GB")

st.sidebar.markdown("---")

# Modellval
model_options = get_available_models()
if model_options:
    selected_model = st.sidebar.selectbox("Välj LLM-modell från KINGSTON", model_options)
    model_full_path = os.path.join(BASE_PATH, "models", selected_model)
else:
    st.sidebar.error("Inga modeller hittades i /Librarian/models/")
    selected_model = None

# Knappar för laddning och rensning
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🚀 Ladda Modeller"):
        if selected_model:
            with st.status("Laddar in i M4 RAM...", expanded=False) as status:
                st.session_state.model, st.session_state.tokenizer = load_llm(model_full_path)
                # Stig till din standard-encoder
                encoder_path = os.path.join(BASE_PATH, "models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
                st.session_state.encoder = load_encoder(encoder_path)
                status.update(label="Modeller redo!", state="complete")
            st.rerun()

with col2:
    if st.button("🗑️ Töm RAM"):
        st.cache_resource.clear()
        if 'model' in st.session_state: del st.session_state.model
        st.sidebar.warning("Cachen rensad.")
        st.rerun()

# DEN RÖDA VARNINGEN
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="color:#FF4B4B; font-weight:bold; font-size:14px; background-color: rgba(255,75,75,0.1); padding:10px; border-radius:5px;">'
    '⚠️ VIKTIGT:<br>Stäng terminalfönstret eller tryck Ctrl + C när du är klar för att frigöra RAM-minnet på din Mac.'
    '</p>', 
    unsafe_allow_html=True
)

# --- HUVUDYTA: TABS ---
tab1, tab2 = st.tabs(["📥 Arkivera (Ingest)", "💬 Bibliotekarien (Chatt)"])

# --- TAB 1: ARKIVERING (DROP-IN) ---
with tab1:
    st.header("Arkivera dokument till user_data")
    st.write("Dra in filer här för att låta AI:n läsa, kategorisera och spara dem i siffer-systemet.")
    
    uploaded_files = st.file_uploader("Släpp filer här (PDF, EPUB, DOCX, TXT)", 
                                    type=['pdf', 'epub', 'docx', 'txt', 'rtf'], 
                                    accept_multiple_files=True)
    
    if uploaded_files:
        if 'model' in st.session_state:
            if st.button("Starta AI-Arkivering"):
                progress_bar = st.progress(0)
                for i, uploaded_file in enumerate(uploaded_files):
                    # Spara filen till raw_data
                    save_path = os.path.join(BASE_PATH, "raw_data", uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.write(f"⚙️ Analyserar: {uploaded_file.name}...")
                    
                    # HÄR ANROPAR DU DIN ARKIVERINGSLOGIK
                    # Exempel: run_archiver_logic(save_path, st.session_state.model, ...)
                    
                    time.sleep(1) # Simulerad tid för test
                    progress_bar.progress((i + 1) / len(uploaded_files))
                st.success(f"Färdig! {len(uploaded_files)} filer har lagts till i arkivet.")
        else:
            st.error("Du måste ladda en modell i sidebaren först för att AI:n ska kunna analysera filerna.")

# --- TAB 2: CHATT ---
with tab2:
    st.header("Konversera med ditt arkiv")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Visa chatt-historik
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Inputfält
    if prompt := st.chat_input("Vad vill du veta?"):
        if 'model' in st.session_state:
            # Spara och visa användarens fråga
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generera svar
            with st.chat_message("assistant"):
                with st.spinner("Söker och tänker..."):
                    # HÄR KOPPLAR DU PÅ DIN SÖKLOGIK
                    # response = generate_librarian_response(prompt, st.session_state.model, ...)
                    response = "Jag är redo! (Här kommer svaret från din smart_search när vi kopplat ihop dem)."
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Ladda modellen i sidebaren innan du kan chatta.")