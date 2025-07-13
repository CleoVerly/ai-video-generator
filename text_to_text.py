# app.py

import streamlit as st
import os
import faiss
import numpy as np
import requests
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ======================================================================================
# Konfigurasi dan Pemuatan Model
# ======================================================================================

# Muat variabel dari file .env
load_dotenv()

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Chatbot EPUB", layout="wide")

# Ambil API key dari environment variables
# Pastikan Anda memiliki file .env dengan OPENROUTER_API_KEY
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("API Key OpenRouter tidak ditemukan. Harap buat file .env dan atur OPENROUTER_API_KEY di dalamnya.")
    st.stop()

@st.cache_resource
def load_embedding_model():
    """Memuat model SentenceTransformer. Dicache agar hanya dimuat sekali."""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ======================================================================================
# Fungsi Pemrosesan EPUB
# ======================================================================================

def extract_text_from_epub(epub_path):
    """Mengekstrak teks mentah dari file EPUB."""
    try:
        book = epub.read_epub(epub_path)
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        full_text = ""
        for item in items:
            # Dapatkan konten dari item
            content = item.get_body_content()
            # PERBAIKAN: Cek apakah konten ada sebelum diproses untuk menghindari error
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                # Hapus tag yang tidak relevan dan ekstrak teksnya
                text = soup.get_text(separator=' ', strip=True)
                full_text += text + " "
        return full_text
    except Exception as e:
        # Menampilkan pesan error yang lebih spesifik jika terjadi masalah lain
        st.error(f"Gagal membaca atau memproses file EPUB '{os.path.basename(epub_path)}': {e}")
        return None

@st.cache_resource(show_spinner="Memproses EPUB: Mengekstrak teks, membuat embedding, dan membangun indeks...")
def process_epub(epub_path):
    """
    Memproses satu file EPUB:
    1. Ekstrak teks.
    2. Pecah teks menjadi potongan (chunks).
    3. Buat embedding untuk setiap chunk.
    4. Bangun indeks FAISS untuk pencarian cepat.
    Fungsi ini di-cache, jadi pemrosesan hanya terjadi sekali per file.
    """
    # 1. Ekstrak Teks
    st.write(f"Mengekstrak teks dari {os.path.basename(epub_path)}...")
    text = extract_text_from_epub(epub_path)
    if not text:
        # Pesan ini akan muncul jika extract_text_from_epub gagal atau tidak menghasilkan teks
        st.error(f"Tidak ada teks yang bisa diekstrak dari file {os.path.basename(epub_path)}. File mungkin rusak atau kosong.")
        return None, None

    # 2. Pecah Teks
    st.write("Memecah teks menjadi potongan yang lebih kecil...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    if not chunks:
        st.warning("Tidak ada teks yang dapat diproses dari file EPUB ini.")
        return None, None

    # 3. Buat Embeddings
    st.write("Membuat embedding (representasi vektor) untuk setiap potongan teks...")
    model = load_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=True).astype('float32')

    # 4. Bangun Indeks FAISS
    st.write("Membangun indeks FAISS untuk pencarian cepat...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    st.success(f"Pemrosesan untuk {os.path.basename(epub_path)} selesai!")
    return index, chunks

# ======================================================================================
# Fungsi Chatbot (RAG: Retrieval-Augmented Generation)
# ======================================================================================

def find_relevant_chunks(query, index, chunks, model, top_k=5): # PERUBAHAN: Nilai default diubah ke 5
    """Mencari potongan teks (chunks) yang paling relevan dengan pertanyaan."""
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    
    # Mengambil teks dari chunk yang relevan
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def get_llm_response(query, context, api_key):
    """Mengirim pertanyaan dan konteks ke LLM untuk mendapatkan jawaban."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Prompt yang dirancang untuk memberikan jawaban berdasarkan konteks
    prompt = (
        "Anda adalah asisten AI yang ahli dalam menganalisis isi buku. "
        "Berdasarkan konteks yang diberikan di bawah ini, jawablah pertanyaan pengguna dengan jelas dan informatif dalam Bahasa Indonesia. "
        "Jika informasi tidak ditemukan dalam konteks, katakan bahwa Anda tidak dapat menemukan jawabannya di dalam buku ini.\n\n"
        f"KONTEKS:\n{' '.join(context)}\n\n"
        f"PERTANYAAN PENGGUNA:\n{query}\n\n"
        "JAWABAN:"
    )
    
    data = {
        "model": "openai/gpt-4o-mini", # Model yang cepat dan cerdas
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3, # Suhu rendah untuk jawaban yang lebih fokus
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=90)
        response.raise_for_status() # Akan error jika status code bukan 2xx
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"Terjadi kesalahan saat menghubungi API OpenRouter: {e}")
        return "Maaf, terjadi kesalahan saat mencoba mendapatkan jawaban."

# ======================================================================================
# Antarmuka Pengguna (UI) Streamlit
# ======================================================================================

st.title("🤖 Chatbot Cerdas Berbasis File EPUB")
st.markdown("Pilih sebuah buku dari koleksi Anda, proses, dan mulailah bertanya!")

# --- SIDEBAR UNTUK KONTROL ---
st.sidebar.title("📚 Koleksi Buku Anda")
st.sidebar.markdown("Letakkan file EPUB Anda di dalam folder `epub_files`.")

# Mencari file EPUB di direktori yang ditentukan
epub_dir = "epub_files"
if not os.path.exists(epub_dir):
    os.makedirs(epub_dir)
    
epub_files = [f for f in os.listdir(epub_dir) if f.endswith('.epub')]

if not epub_files:
    st.sidebar.warning("Tidak ada file .epub yang ditemukan di folder `epub_files`.")
    st.stop()

# Dropdown untuk memilih file EPUB
selected_epub = st.sidebar.selectbox(
    "Pilih file EPUB untuk dijadikan basis data:",
    epub_files
)

# Tombol untuk memulai pemrosesan
if st.sidebar.button("Proses File EPUB Pilihan", type="primary"):
    epub_path = os.path.join(epub_dir, selected_epub)
    # Memanggil fungsi pemrosesan
    index, chunks = process_epub(epub_path)
    # Hanya lanjut jika pemrosesan berhasil
    if index is not None and chunks is not None:
        # Menyimpan data yang telah diproses ke session_state
        st.session_state.processed_data = {
            "file_name": selected_epub,
            "index": index,
            "chunks": chunks
        }
        # Mengosongkan histori chat jika file baru diproses
        st.session_state.messages = []
        # Memaksa refresh UI untuk menampilkan status baru
        st.rerun()

# Menampilkan status file yang sedang aktif
if 'processed_data' in st.session_state:
    st.sidebar.success(f"✅ Aktif: **{st.session_state.processed_data['file_name']}**")
    st.sidebar.info("Anda sekarang dapat memulai percakapan dengan chatbot.")
else:
    st.sidebar.info("Silakan pilih file dan klik tombol proses untuk memulai.")

# --- AREA CHAT UTAMA ---

# Inisialisasi histori chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan pesan-pesan dari histori
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # PERUBAHAN: Tampilkan expander sumber jika pesan dari asisten dan memiliki konteks
        if message["role"] == "assistant" and "context" in message and message["context"]:
            with st.expander("Lihat Sumber Asli dari EPUB"):
                # Gabungkan semua potongan teks sumber dan tampilkan
                source_text = "\n\n---\n\n".join(message["context"])
                st.markdown(f"_{source_text}_")


# Input dari pengguna
if prompt := st.chat_input("Tanyakan sesuatu tentang isi buku ini..."):
    # Cek apakah sudah ada data yang diproses
    if 'processed_data' not in st.session_state:
        st.warning("Harap proses file EPUB terlebih dahulu melalui panel di sebelah kiri.")
        st.stop()

    # Tambahkan pesan pengguna ke histori dan tampilkan
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Dapatkan data yang relevan dari session state
    model = load_embedding_model()
    index = st.session_state.processed_data['index']
    chunks = st.session_state.processed_data['chunks']

    # Tampilkan pesan "thinking..."
    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi dan berpikir..."):
            # 1. Cari konteks yang relevan
            # PERUBAHAN: Secara eksplisit memanggil dengan top_k=5 untuk pencarian yang lebih luas
            relevant_context = find_relevant_chunks(prompt, index, chunks, model, top_k=5)
            
            # 2. Dapatkan jawaban dari LLM
            response = get_llm_response(prompt, relevant_context, OPENROUTER_API_KEY)
            
            # 3. Tampilkan jawaban AI
            st.markdown(response)

            # PERUBAHAN: Tambahkan expander sumber di bawah jawaban yang baru dibuat
            with st.expander("Lihat Sumber Asli dari EPUB"):
                source_text = "\n\n---\n\n".join(relevant_context)
                st.markdown(f"_{source_text}_")
    
    # PERUBAHAN: Tambahkan jawaban AI DAN konteksnya ke histori
    st.session_state.messages.append({"role": "assistant", "content": response, "context": relevant_context})
