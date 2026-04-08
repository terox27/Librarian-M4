# Librarian AI

Modular AI Long-Term Memory System

## Features

- Supports multiple LLM models including Gemma 4, Llama 3.1, Phi-3.5, and Qwen2.5
- Vector-based search using multilingual embeddings
- Document processing (PDF, DOCX, EPUB, RTF, TXT)
- Web interface via Streamlit
- Terminal interface for direct interaction

## Supported Models

The system supports various quantized models from the `models/` directory:

- **Gemma 4** (recommended): `gemma-4-e4b-it-nvfp4`, `gemma-4-e4b-it-mxfp4`, etc.
- **Llama 3.1**: `Llama-3.1-8B-8bit`
- **Phi-3.5**: `Phi-3.5-mini-instruct-4bit`, `Phi-3.5-mini-instruct-8bit`
- **Qwen2.5**: `Qwen2.5-Coder-7B-Instruct-4bit`, `Qwen2.5-Coder-7B-Instruct-8bit`

## Usage

### Web Interface
```bash
streamlit run web_app.py
```

### Terminal Interface
```bash
python mac_librarian.py
```

### Requirements
- Python 3.12+
- MLX (Apple Silicon)
- Dependencies in requirements.txt
