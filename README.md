# 📚 RAG with Gradio — Per-user Storage, OpenAI, FAISS  

This repository provides a **Retrieval-Augmented Generation (RAG) system** built with **Gradio**, **OpenAI embeddings & ChatCompletion API**, and **FAISS vector search**.  
It supports **multi-user login/registration**, **per-user storage**, **document uploads**, and **chat history tracking**.  

---

## 🚀 Features  

- 🔐 **User Authentication**
  - Secure login & registration using salted+hashed passwords (PBKDF2-HMAC-SHA256).  
  - Per-user storage isolation.  

- 📂 **Document Upload & Processing**
  - Supports **PDF, TXT, DOCX**.  
  - Streaming-based text extraction (avoids memory overflow on large files).  
  - Sliding-window **chunking with overlap**.  
  - Per-file safety limit (`MAX_CHARS_PER_FILE`) to prevent huge uploads.  

- 🧠 **RAG Pipeline**
  - Embeddings with **OpenAI’s `text-embedding-3-small`**.  
  - Document indexing & retrieval with **FAISS (L2 search)**.  
  - Context-aware answering with **OpenAI ChatCompletion (`gpt-3.5-turbo`)**.  
  - Automatic **source citation (filename)** in responses.  

- 💾 **Persistence**
  - SQLite database for **users, uploads, chat history**.  
  - FAISS + JSON metadata for vector indexes.  

- 🖥️ **Gradio Interface**
  - Tabs for:
    - Login / Register / Logout  
    - Document Upload  
    - Query + Chat History Viewer  
  - Real-time answer generation with retrieval augmentation.  

---

## ⚙️ System Design  
<img width="680" height="626" alt="image" src="https://github.com/user-attachments/assets/685ab07b-5d0d-43fb-9d56-fc4ef4c48e62" />

## Project Structure
<img width="523" height="175" alt="image" src="https://github.com/user-attachments/assets/5aba3552-80c9-4d07-a1c4-6577f7ce35a8" />

## Screenshots
