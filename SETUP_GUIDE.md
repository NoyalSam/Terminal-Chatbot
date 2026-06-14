# Setup Guide

This guide explains how to run the `rag-genai-project` locally.

## 1. Prerequisites

Make sure the following are installed:

- Python 3.10 or newer
- `pip`
- A Google AI API key

This project currently uses:

- `langchain`
- `langchain-community`
- `langchain-google-genai`
- `faiss-cpu`
- `pypdf`
- `python-dotenv`
- `sentence-transformers`

## 2. Open the Project Folder

From a terminal:

```powershell
cd c:\Users\Administrator\Downloads\rag-genai-project\rag-genai-project
```

## 3. Create and Activate a Virtual Environment

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env\Scripts\Activate.ps1
```

## 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 5. Configure the API Key

The application expects `GOOGLE_API_KEY` to be available.

Create a `.env` file in the project root with:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Notes:

- `config.py` loads environment variables with `python-dotenv`
- `config_validator.py` checks that `GOOGLE_API_KEY` exists before startup

## 6. Add the Input Document

The current `app.py` loads this file by default:

```text
data/sample_document.txt
```

Replace that file with your own `.txt` content, or update the path in `app.py` if you want to use a different `.txt` or `.pdf` file.

Supported formats:

- `.txt`
- `.pdf`

## 7. Run the Application

```powershell
python app.py
```

If startup is successful, you should see messages similar to:

```text
Configuration validated successfully.
Loading documents...
Chunking documents...
Creating vector store...
RAG Agent Ready
```

## 8. Use the Chat Loop

Once the app is running, type your questions at the prompt:

```text
User: What is this document about?
```

To stop the program, type:

```text
exit
```

## 9. Common Issues

### `GOOGLE_API_KEY missing in environment variables`

Add your API key to `.env` and restart the app.

### Module import errors

Make sure the virtual environment is activated before running `python app.py`.

### PowerShell activation blocked

Use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

then activate the environment again.

### FAISS or dependency install problems

Upgrade packaging tools first:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 10. Current Behavior Summary

The current implementation:

- loads `data/sample_document.txt`
- uses Google embeddings
- uses the `gemini-2.5-flash` chat model
- builds a FAISS vector store
- starts a terminal-based RAG agent with memory

## 11. Recommended Next Improvement

For safer configuration, move all secrets fully into `.env` and avoid storing API keys directly in source files.
