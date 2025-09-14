"""
RAG with Gradio — per-user storage, OpenAI embeddings & ChatCompletion, FAISS vector index.

Fixes included:
- streaming/batched chunking for uploads (prevents MemoryError)
- file path handling for Gradio's upload behaviour
- limits on maximum characters processed per file
"""

import os
import json
import time
import faiss
import sqlite3
import secrets
import hashlib
import threading
from typing import List, Tuple, Iterable
from datetime import datetime
import numpy as np
import gradio as gr
from openai import OpenAI
from PyPDF2 import PdfReader
try:
    import docx
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False
from dotenv import load_dotenv
load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

client = OpenAI()

# --------------------------- CONFIG ---------------------------
# openai.api_key = OPENAI_API_KEY

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

# Safety limits (adjust as needed)
MAX_CHARS_PER_FILE = 300_000          # maximum characters to process per uploaded file
EMBED_BATCH_SIZE = 32                 # how many chunks to embed per API call

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
DB_PATH = os.path.join(BASE_DIR, "users.db")
os.makedirs(USER_DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

DB_LOCK = threading.Lock()
INDEX_LOCK = threading.Lock()

# --------------------------- DB HELPERS ---------------------------

def init_db():
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            question TEXT,
            answer TEXT,
            ts TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """)
        conn.commit()
        conn.close()

init_db()

# --------------------------- AUTH HELPERS ---------------------------

def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200_000)
    return dk.hex()


def create_user(username: str, password: str) -> Tuple[bool, str]:
    salt = secrets.token_bytes(16)
    pwd_hash = _hash_password(password, salt)
    created_at = datetime.utcnow().isoformat()
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, password_salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
                        (username, salt.hex(), pwd_hash, created_at))
            conn.commit()
            user_id = cur.lastrowid
            conn.close()
            os.makedirs(os.path.join(USER_DATA_DIR, str(user_id)), exist_ok=True)
            return True, "User created"
        except sqlite3.IntegrityError:
            conn.close()
            return False, "Username already exists"


def verify_user(username: str, password: str) -> Tuple[bool, int]:
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, password_salt, password_hash FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()
    if not row:
        return False, -1
    user_id, salt_hex, stored_hash = row
    salt = bytes.fromhex(salt_hex)
    check_hash = _hash_password(password, salt)
    return (check_hash == stored_hash), user_id

# --------------------------- DOC PROCESSING ---------------------------

def extract_text_from_pdf_stream(path: str) -> Iterable[str]:
    """
    Generator yields text per page. This avoids building a giant string in memory.
    """
    reader = PdfReader(path)
    for p in reader.pages:
        try:
            t = p.extract_text()
            if t:
                yield t
        except Exception:
            continue

def extract_text_from_docx_stream(path: str) -> Iterable[str]:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx not installed")
    doc = docx.Document(path)
    # yield paragraphs in small batches
    for p in doc.paragraphs:
        if p.text:
            yield p.text

def extract_text_from_txt_stream(path: str) -> Iterable[str]:
    # read file in streaming manner (line by line)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line:
                yield line

def sliding_chunk_generator(text_iter: Iterable[str], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, max_chars=MAX_CHARS_PER_FILE):
    """
    Consume an iterable of text pieces (pages/paragraphs/lines) and yield chunks (strings).
    This avoids loading the entire document at once.
    Also enforces an overall max character cap.
    """
    buffer = ""
    total_consumed = 0
    for piece in text_iter:
        if total_consumed >= max_chars:
            break
        needed = max_chars - total_consumed
        # clip piece if it would exceed overall allowed chars
        if len(piece) > needed:
            piece = piece[:needed]
        buffer += ("\n" + piece) if buffer else piece
        total_consumed += len(piece)
        # emit chunks while buffer is large enough
        while len(buffer) >= chunk_size:
            chunk = buffer[:chunk_size]
            yield chunk
            # remove chunk but keep overlap
            buffer = buffer[chunk_size - overlap:]
    # after iteration, if buffer has remainder, yield it too (if non-empty)
    if buffer.strip():
        yield buffer

# --------------------------- OPENAI / FAISS HELPERS ---------------------------

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def _index_meta_path(user_id: int) -> str:
    return os.path.join(INDEX_DIR, f"{user_id}_meta.json")

def _index_path(user_id: int) -> str:
    return os.path.join(INDEX_DIR, f"{user_id}.faiss")

def load_user_index(user_id: int):
    path = _index_path(user_id)
    meta_path = _index_meta_path(user_id)
    if not os.path.exists(path) or not os.path.exists(meta_path):
        return None, {}
    idx = faiss.read_index(path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return idx, meta

def save_user_index(user_id: int, index, meta: dict):
    path = _index_path(user_id)
    meta_path = _index_meta_path(user_id)
    faiss.write_index(index, path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def ensure_index_for_user(user_id: int, dim: int):
    idx, meta = load_user_index(user_id)
    if idx is None:
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(quantizer)
        meta = {}
        save_user_index(user_id, index, meta)
        return index, meta
    return idx, meta

import faiss
import os
import json

def load_index(user_id: int):
    """
    Load FAISS index + metadata for a user.
    If none exists, create a new index and empty metadata.
    """
    index_path = os.path.join(INDEX_DIR, f"{user_id}.index")
    meta_path = os.path.join(INDEX_DIR, f"{user_id}_meta.json")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            metas = json.load(f)
    else:
        index = faiss.IndexFlatL2(1536)   # ✅ dimension must match embedding size (1536 for text-embedding-3-small)
        metas = []
    return index, metas

def save_index(user_id: int, index, metas):
    """
    Save FAISS index + metadata for a user.
    """
    index_path = os.path.join(INDEX_DIR, f"{user_id}.index")
    meta_path = os.path.join(INDEX_DIR, f"{user_id}_meta.json")

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metas, f)

def add_documents_to_index(user_id: int, docs: List[Tuple[str, str]]) -> int:
    """
    Add a list of (chunk_text, filename) docs to the FAISS index for this user.
    Returns the number of chunks successfully added.
    """
    index, metas = load_index(user_id)   # load FAISS + metadata
    xb = []
    added = 0

    for chunk, filename in docs:
        if not chunk.strip():
            continue
        try:
            vec = get_embedding(chunk)   # ✅ fixed for new OpenAI SDK
            xb.append(vec)
            metas.append({"text": chunk, "filename": filename})
            added += 1
        except Exception as e:
            print(f"[ERROR] Embedding error for {filename}: {e}")

    if xb:
        xb = np.array(xb).astype("float32")
        with INDEX_LOCK:   # prevent race conditions
            index.add(xb)
            save_index(user_id, index, metas)

    return added


def query_index(user_id: int, query: str, top_k=TOP_K):
    emb = get_embedding(query)
    with INDEX_LOCK:
        idx, meta = load_index(user_id)
        if idx is None:
            return []
        import numpy as np
        vec = np.array([emb]).astype('float32')
        D, I = idx.search(vec, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:  
                continue
            if idx < len(meta):
                data = meta[idx]
                results.append({
                    "id": int(idx),
                    "score": float(score),
                    "text": data.get("text"),
                    "source": data.get("filename")
                })
        return results

# --------------------------- FILE / DB RECORDS ---------------------------

def save_uploaded_file(user_id: int, filename: str, file_path: str) -> str:
    user_folder = os.path.join(USER_DATA_DIR, str(user_id))
    os.makedirs(user_folder, exist_ok=True)
    safe_name = filename.replace('..', '')
    new_path = os.path.join(user_folder, f"{int(time.time())}_{safe_name}")
    with open(file_path, 'rb') as fsrc, open(new_path, 'wb') as fdst:
        fdst.write(fsrc.read())
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO documents (user_id, filename, filepath, uploaded_at) VALUES (?, ?, ?, ?)",
                    (user_id, filename, new_path, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    return new_path

def process_file_and_index(user_id: int, filepath: str, filename: str) -> Tuple[int, str]:
    """
    Stream the file, generate chunks, and add them to the index in batches.
    Returns (number_of_chunks_indexed, status_message)
    """
    try:
        lower = filename.lower()
        if lower.endswith('.pdf'):
            text_iter = extract_text_from_pdf_stream(filepath)
        elif lower.endswith('.docx') and DOCX_AVAILABLE:
            text_iter = extract_text_from_docx_stream(filepath)
        else:
            text_iter = extract_text_from_txt_stream(filepath)

        # generate chunks lazily
        chunk_gen = sliding_chunk_generator(text_iter, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHARS_PER_FILE)

        # batch up chunks and add in groups to avoid storing all at once
        batch = []
        total_added = 0
        batch_src = filename
        for chunk in chunk_gen:
            if not chunk.strip():
                continue
            batch.append((chunk, batch_src))
            if len(batch) >= EMBED_BATCH_SIZE:
                added = add_documents_to_index(user_id, batch)
                total_added += added
                batch = []
        # final flush
        if batch:
            added = add_documents_to_index(user_id, batch)
            total_added += added

        # if we truncated due to MAX_CHARS_PER_FILE, warn user:
        truncated_warning = ""
        # We can check file size in characters by re-opening briefly if needed; instead we rely on how many chunks added.
        return total_added, f"Indexed {total_added} chunks from {filename}. {truncated_warning}"
    except MemoryError:
        return 0, "MemoryError while processing file — file may be too large. Increase system memory or lower MAX_CHARS_PER_FILE."
    except Exception as e:
        return 0, f"Error processing file {filename}: {str(e)}"

# --------------------------- RAG ANSWER ---------------------------

def generate_answer(user_id: int, question: str) -> str:
    hits = query_index(user_id, question, top_k=TOP_K)
    if not hits:
        prompt = f"Answer the question based only on your knowledge. If you don't know, say you don't know.\n\nQuestion:\n{question}\n"
    else:
        context = "\n\n---\n\n".join([f"Source: {h['source']}\n{h['text']}" for h in hits])
        prompt = f"You are a helpful assistant. Use the following extracted context from the user's documents to answer the question. Cite the source filename when possible.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer concisely."

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    answer = resp.choices[0].message.content.strip()
    # store chat
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO chats (user_id, question, answer, ts) VALUES (?, ?, ?, ?)",
                    (user_id, question, answer, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    return answer

# --------------------------- GRADIO APP ---------------------------

with gr.Blocks() as demo:
    state = gr.State({'user_id': None, 'username': None})

    with gr.Tab("Login / Register"):
        gr.Markdown("# RAG - Login or Register")
        username_in = gr.Textbox(label="Username")
        password_in = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        register_btn = gr.Button("Register")
        logout_btn = gr.Button("Logout")
        login_message = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("Uploader"):
        gr.Markdown("# Upload documents (login required)")
        # Gradio's File component returns a list of file paths by default (type='file' or None)
        upload_files = gr.File(file_types=[".pdf", ".txt", ".docx"], file_count="multiple", type="filepath", label="Upload PDF / TXT / DOCX")
        upload_btn = gr.Button("Upload")
        upload_status = gr.Textbox(label="Upload status", interactive=False)

    with gr.Tab("Query"):
        gr.Markdown("# Ask questions (login required)")
        question_box = gr.Textbox(label="Question")
        ask_btn = gr.Button("Ask")
        answer_out = gr.Textbox(label="Answer", interactive=False)
        history_dropdown = gr.Dropdown(choices=[], label="Chat history (latest first)", allow_custom_value=True)

    # helper callbacks
    def _on_register(username, password, state_obj):
        ok, msg = create_user(username, password)
        return msg, state_obj

    def _on_login(username, password, state_obj):
        ok, user_id = verify_user(username, password)
        if ok:
            state_obj['user_id'] = user_id
            state_obj['username'] = username
            return f"Logged in as {username}", state_obj
        else:
            return "Invalid credentials", state_obj

    def _on_upload(files, state_obj):
        """
        files: list of file path strings (Gradio returns path strings)
        streaming + batching is used to avoid MemoryError
        """
        if not state_obj.get('user_id'):
            return "Not logged in"
        user_id = state_obj['user_id']
        if not files:
            return "No files"
        total_chunks = 0
        messages = []
        for file_path in files:
            # file_path is the temporary path Gradio gives us
            if not file_path:
                continue
            filename = os.path.basename(file_path)
            try:
                new_path = save_uploaded_file(user_id, filename, file_path)
            except Exception as e:
                messages.append(f"Failed to save {filename}: {e}")
                continue
            # process & index streamingly
            added, status = process_file_and_index(user_id, new_path, filename)
            total_chunks += added
            messages.append(status)
        return f"Uploaded {len(files)} files, indexed {total_chunks} chunks. Details: {'; '.join(messages)}"

    def _on_ask(question, state_obj):
        if not state_obj.get('user_id'):
            return "Please login first", gr.update(choices=[])
        user_id = state_obj['user_id']
        ans = generate_answer(user_id, question)
        # refresh history
        history = _get_history(user_id)
        return ans, gr.update(choices=history, value=None)

    def _get_history(user_id):
        with DB_LOCK:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT id, question, answer, ts FROM chats WHERE user_id = ? ORDER BY id DESC LIMIT 50", (user_id,))
            rows = cur.fetchall()
            conn.close()
        choices = [f"{r[0]} - {r[1][:80]}... ({r[3]})" for r in rows]
        return choices
    
    def _on_logout(state_obj):
        state_obj['user_id'] = None
        state_obj['username'] = None
        return "Logged out successfully", state_obj

    # wire events
    register_btn.click(_on_register, inputs=[username_in, password_in, state], outputs=[login_message, state])
    login_btn.click(_on_login, inputs=[username_in, password_in, state], outputs=[login_message, state])
    upload_btn.click(_on_upload, inputs=[upload_files, state], outputs=[upload_status])
    ask_btn.click(_on_ask, inputs=[question_box, state], outputs=[answer_out, history_dropdown])
    logout_btn.click(_on_logout, inputs=[state], outputs=[login_message, state])

    # allow selecting a history item to view full answer
    def view_history(item_str, state_obj):
        if not item_str:
            return ""
        chat_id = int(item_str.split(' - ')[0])
        with DB_LOCK:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT question, answer, ts FROM chats WHERE id = ?", (chat_id,))
            row = cur.fetchone()
            conn.close()
        if not row:
            return ""
        q, a, ts = row
        return f"{ts}\nQ: {q}\n\nA: {a}"

    history_dropdown.change(view_history, inputs=[history_dropdown, state], outputs=[answer_out])

if __name__ == '__main__':
    demo.launch(share=False)
