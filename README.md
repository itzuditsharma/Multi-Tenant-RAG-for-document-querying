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
- Register User
  <img width="1902" height="701" alt="image" src="https://github.com/user-attachments/assets/5a799cc0-9976-4365-9f3a-9af0d563b581" />
- Login
  <img width="1871" height="727" alt="image" src="https://github.com/user-attachments/assets/aae25468-afcb-4771-86ec-900a900a28b2" />
  <img width="1860" height="740" alt="image" src="https://github.com/user-attachments/assets/48316667-0f45-49eb-9b43-73034b03a649" />
- Ingestion
  Udit
  <img width="1879" height="490" alt="image" src="https://github.com/user-attachments/assets/de1313d0-b351-4a24-bf8f-468e4ec1cd23" />
  John
  <img width="1834" height="539" alt="image" src="https://github.com/user-attachments/assets/8df46d86-536d-40c3-9018-f8dad0c6f072" />
- Retreival
  Udit: 
  <img width="1866" height="629" alt="image" src="https://github.com/user-attachments/assets/56825a0b-0f74-4892-bec5-5f9e757d3609" />
  <img width="1861" height="652" alt="image" src="https://github.com/user-attachments/assets/5fb21931-f619-4249-93b3-67bac32c2012" />
  John:
  <img width="1857" height="636" alt="image" src="https://github.com/user-attachments/assets/729f7749-ee2a-4c4f-a62a-822b33f58ce3" />
  <img width="1841" height="598" alt="image" src="https://github.com/user-attachments/assets/493e5154-cdea-4ec3-8208-6a8a12c08646" />


  





