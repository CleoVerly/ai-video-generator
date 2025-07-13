# app.py

import streamlit as st
import os
import faiss
import numpy as np
import requests
import re
import hashlib
import textwrap
from io import BytesIO

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips

# ======================================================================================
# Konfigurasi dan Pemuatan Model
# ======================================================================================

# Muat variabel dari file .env
load_dotenv()

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Chatbot EPUB", layout="wide")

# Ambil API key dari environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY") # Diperlukan untuk video

if not OPENROUTER_API_KEY:
    st.error("API Key OpenRouter tidak ditemukan. Harap atur di file .env Anda.")
    st.stop()
# Peringatan jika Pexels API Key tidak ada, karena ini fitur opsional
if not PEXELS_API_KEY:
    st.sidebar.warning("Pexels API Key tidak ditemukan di file .env. Fitur pembuatan video tidak akan berfungsi.")


@st.cache_resource
def load_embedding_model():
    """Memuat model SentenceTransformer. Dicache agar hanya dimuat sekali."""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ======================================================================================
# Fungsi Pemrosesan EPUB (Logika Chatbot)
# ======================================================================================

def extract_text_from_epub(epub_path):
    """Mengekstrak teks mentah dari file EPUB."""
    try:
        book = epub.read_epub(epub_path)
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        full_text = ""
        for item in items:
            content = item.get_body_content()
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                full_text += text + " "
        return full_text
    except Exception as e:
        st.error(f"Gagal membaca file EPUB '{os.path.basename(epub_path)}': {e}")
        return None

@st.cache_resource(show_spinner="Memproses EPUB: Mengekstrak teks, membuat embedding, dan membangun indeks...")
def process_epub(epub_path):
    """Memproses satu file EPUB untuk RAG."""
    text = extract_text_from_epub(epub_path)
    if not text:
        st.error(f"Tidak ada teks yang bisa diekstrak dari {os.path.basename(epub_path)}.")
        return None, None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    if not chunks:
        st.warning("Tidak ada teks yang dapat diproses dari file EPUB ini.")
        return None, None

    model = load_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=True).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    st.success(f"Pemrosesan untuk {os.path.basename(epub_path)} selesai!")
    return index, chunks

# ======================================================================================
# Fungsi Chatbot (RAG: Retrieval-Augmented Generation)
# ======================================================================================

def find_relevant_chunks(query, index, chunks, model, top_k=5):
    """Mencari potongan teks yang paling relevan."""
    query_embedding = model.encode([query]).astype('float32')
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def get_llm_response(query, context, api_key):
    """Mendapatkan jawaban dari LLM berdasarkan konteks."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = (
        "Anda adalah asisten AI yang ahli menganalisis isi buku. "
        "Berdasarkan konteks berikut, jawab pertanyaan pengguna dalam Bahasa Indonesia. "
        "Jika informasi tidak ada, katakan Anda tidak dapat menemukannya di buku ini.\n\n"
        f"KONTEKS:\n{' '.join(context)}\n\n"
        f"PERTANYAAN:\n{query}\n\nJAWABAN:"
    )
    data = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=90)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"Error menghubungi API OpenRouter: {e}")
        return "Maaf, terjadi kesalahan saat mencoba mendapatkan jawaban."

# ======================================================================================
# Fungsi Pembuatan Video
# ======================================================================================

def text_to_speech_gtts(text, output_path, lang='id'):
    """Mengonversi teks ke audio MP3."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        return output_path
    except Exception as e:
        st.error(f"Gagal membuat file audio: {e}")
        return None

def download_image_from_pexels(query, api_key):
    """Mengunduh satu gambar dari Pexels."""
    if not api_key: return None
    headers = {"Authorization": api_key}
    try:
        res = requests.get(f"https://api.pexels.com/v1/search?query={query}&per_page=1", headers=headers)
        res.raise_for_status()
        data = res.json()
        if data["photos"]:
            img_url = data["photos"][0]["src"]["landscape"]
            img_data = requests.get(img_url).content
            return Image.open(BytesIO(img_data)).resize((1280, 720))
    except Exception:
        return None
    return None

def create_text_image(text, size=(1280, 720)):
    """Membuat gambar dengan teks di tengahnya."""
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=60)
    except IOError:
        font = ImageFont.load_default()
    
    wrapped_text = textwrap.fill(text, width=30)
    _, _, text_w, text_h = draw.textbbox((0, 0), wrapped_text, font=font, align="center")
    x = (size[0] - text_w) / 2
    y = (size[1] - text_h) / 2
    
    # Efek outline sederhana
    stroke_width = 2
    for pos in [ (x-stroke_width, y), (x+stroke_width, y), (x, y-stroke_width), (x, y+stroke_width) ]:
        draw.text(pos, wrapped_text, font=font, fill="black", align="center")

    draw.text((x, y), wrapped_text, font=font, fill="white", align="center")
    return img

def generate_video_from_text(text, pexels_api_key, output_path):
    """Orkestrator utama untuk membuat video dari teks."""
    # 1. Buat Audio
    audio_path = output_path.replace(".mp4", ".mp3")
    if not text_to_speech_gtts(text, audio_path): return None
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    
    # 2. Siapkan klip video
    text_chunks = textwrap.wrap(text, width=80) # Pecah teks untuk adegan berbeda
    if not text_chunks: return None
    
    chunk_duration = duration / len(text_chunks)
    video_clips = []
    
    with st.spinner(f"Membuat video... Mengunduh gambar dan menyusun {len(text_chunks)} adegan."):
        for i, chunk in enumerate(text_chunks):
            # Cari gambar berdasarkan beberapa kata kunci dari chunk
            keyword_query = "islamic " + " ".join(chunk.split()[:4])
            bg_image = download_image_from_pexels(keyword_query, pexels_api_key)
            if bg_image is None: # Fallback ke gambar hitam
                bg_image = Image.new("RGB", (1280, 720), (0, 0, 0))

            background_clip = ImageClip(np.array(bg_image)).set_duration(chunk_duration)
            
            text_image = create_text_image(chunk)
            text_clip = ImageClip(np.array(text_image)).set_duration(chunk_duration)
            
            # Gabungkan background dan teks
            composite_clip = CompositeVideoClip([background_clip, text_clip.set_position("center")])
            video_clips.append(composite_clip)

    # 3. Gabungkan semua klip dan tambahkan audio
    final_video = concatenate_videoclips(video_clips).set_audio(audio_clip)
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", logger=None)
    
    # Hapus file audio sementara
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    return output_path

# ======================================================================================
# Antarmuka Pengguna (UI) Streamlit
# ======================================================================================

st.title("🤖 Chatbot Cerdas Berbasis File EPUB")
st.markdown("Pilih buku, ajukan pertanyaan, dan ubah jawabannya menjadi video!")

# --- SIDEBAR ---
st.sidebar.title("📚 Koleksi Buku Anda")
epub_dir = "epub_files"
if not os.path.exists(epub_dir): os.makedirs(epub_dir)
epub_files = [f for f in os.listdir(epub_dir) if f.endswith('.epub')]

if not epub_files:
    st.sidebar.warning(f"Tidak ada file .epub di folder `{epub_dir}`.")
    st.stop()

selected_epub = st.sidebar.selectbox("Pilih file EPUB:", epub_files)

if st.sidebar.button("Proses File EPUB Pilihan", type="primary"):
    epub_path = os.path.join(epub_dir, selected_epub)
    index, chunks = process_epub(epub_path)
    if index is not None and chunks is not None:
        st.session_state.processed_data = {"file_name": selected_epub, "index": index, "chunks": chunks}
        st.session_state.messages = []
        st.rerun()

if 'processed_data' in st.session_state:
    st.sidebar.success(f"✅ Aktif: **{st.session_state.processed_data['file_name']}**")
else:
    st.sidebar.info("Pilih file dan klik proses untuk memulai.")

# --- AREA CHAT UTAMA ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan histori chat
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            # Tampilkan sumber
            if "context" in message and message["context"]:
                with st.expander("Lihat Sumber Asli dari EPUB"):
                    st.markdown(f"_{'---'.join(message['context'])}_")
            
            # Logika untuk menampilkan video atau tombol pembuatannya
            video_placeholder = st.empty()
            if "video_path" in message and os.path.exists(message["video_path"]):
                with video_placeholder.container():
                    st.video(message["video_path"])
                    with open(message["video_path"], "rb") as f:
                        st.download_button("Unduh Video", f, file_name=os.path.basename(message["video_path"]))
            elif PEXELS_API_KEY: # Hanya tampilkan tombol jika API key ada
                if st.button(f"🎬 Buat Video dari Jawaban Ini", key=f"vid_{i}"):
                    with video_placeholder.container():
                        output_dir = "output_videos"
                        if not os.path.exists(output_dir): os.makedirs(output_dir)
                        file_hash = hashlib.md5(message["content"].encode()).hexdigest()
                        video_path = os.path.join(output_dir, f"video_{file_hash}.mp4")
                        
                        generated_path = generate_video_from_text(message["content"], PEXELS_API_KEY, video_path)
                        
                        if generated_path:
                            st.session_state.messages[i]["video_path"] = generated_path
                            st.rerun()
                        else:
                            st.error("Gagal membuat video.")

# Input dari pengguna
if prompt := st.chat_input("Tanyakan sesuatu tentang isi buku ini..."):
    if 'processed_data' not in st.session_state:
        st.warning("Harap proses file EPUB terlebih dahulu.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    model = load_embedding_model()
    index = st.session_state.processed_data['index']
    chunks = st.session_state.processed_data['chunks']

    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi dan berpikir..."):
            relevant_context = find_relevant_chunks(prompt, index, chunks, model, top_k=5)
            response = get_llm_response(prompt, relevant_context, OPENROUTER_API_KEY)
            
            st.markdown(response)
            with st.expander("Lihat Sumber Asli dari EPUB"):
                st.markdown(f"_{'---'.join(relevant_context)}_")
            
            # Tambahkan pesan ke session state dan rerun untuk menampilkan tombol video
            st.session_state.messages.append({"role": "assistant", "content": response, "context": relevant_context})
            st.rerun()
