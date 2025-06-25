# app.py

import streamlit as st
import json
import os
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from gtts import gTTS
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips
import textwrap
import hashlib
import re
from dotenv import load_dotenv
from googletrans import Translator

# ======================================================================================
# Konfigurasi Aplikasi dan Fungsi Cache
# ======================================================================================

# Muat variabel dari file .env
load_dotenv()

st.set_page_config(page_title="AI Video Generator", layout="wide")

# Ambil konfigurasi dari environment variables
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
pexels_api_key = os.getenv("PEXELS_API_KEY")

@st.cache_resource
def load_model():
    """Memuat model SentenceTransformer dan hanya dijalankan sekali."""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def load_data():
    """Memuat data indeks FAISS, mapping chunk, dan semua chunk teks."""
    faiss_path = "data/faiss_index.bin"
    mapping_path = "data/chunk_mapping.json"
    chunks_path = "data/epub_chunks_translated.json"

    missing_files = []
    if not os.path.exists(faiss_path):
        missing_files.append(faiss_path)
    if not os.path.exists(mapping_path):
        missing_files.append(mapping_path)
    if not os.path.exists(chunks_path):
        missing_files.append(chunks_path)

    if missing_files:
        st.error(
            "Error: File data penting tidak ditemukan.\n\n"
            f"Pastikan file berikut ada di folder `data/`: `{'`, `'.join(missing_files)}`\n\n"
        )
        return None, None, None

    try:
        index = faiss.read_index(faiss_path)
        with open(mapping_path, "r", encoding="utf-8") as f:
            chunk_sources = json.load(f)
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        all_chunks = [chunk for book_title, chunks in data.items() for chunk in chunks if chunk.strip()]
        return index, chunk_sources, all_chunks
    except Exception as e:
        st.error(f"Terjadi error saat memuat data: {e}")
        return None, None, None

# ======================================================================================
# Fungsi Inti (Logika dari Notebook)
# ======================================================================================

def create_text_image_with_pillow(text, size, font_path=None, font_size=90, text_color=(255, 255, 255), bg_color=(0,0,0,0)):
    img = Image.new('RGBA', size, bg_color)
    draw = ImageDraw.Draw(img)

    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    wrapped_text = textwrap.fill(text, width=25)
    
    _, _, text_w, text_h = draw.textbbox((0, 0), wrapped_text, font=font, align="center")
    x = (size[0] - text_w) / 2
    y = (size[1] - text_h) / 2

    stroke_color = "black"
    stroke_width = 3
    draw.text((x-stroke_width, y-stroke_width), wrapped_text, font=font, fill=stroke_color, align="center")
    draw.text((x+stroke_width, y-stroke_width), wrapped_text, font=font, fill=stroke_color, align="center")
    draw.text((x-stroke_width, y+stroke_width), wrapped_text, font=font, fill=stroke_color, align="center")
    draw.text((x+stroke_width, y+stroke_width), wrapped_text, font=font, fill=stroke_color, align="center")

    draw.text((x, y), wrapped_text, font=font, fill=text_color, align="center")
    
    return img

def search_best_chunk(question, model, index, all_chunks, chunk_sources, top_k=1):
    question_embedding = model.encode([question]).astype('float32')
    D, I = index.search(question_embedding, top_k)
    results = []
    for idx in I[0]:
        if idx < len(all_chunks):
            results.append((chunk_sources[idx], all_chunks[idx]))
    return results

def summarize_with_deepseek(text, api_key, output_language="English"):
    """Membuat ringkasan narasi dengan pembersihan teks yang lebih baik."""
    if not api_key:
        st.error("API Key OpenRouter tidak ditemukan. Harap atur di file .env atau secrets Hugging Face.")
        return None
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = { "Authorization": f"Bearer {api_key}" }
    prompt = (f"Write a compelling final narration in {output_language} for a 1-minute YouTube video, "
              f"based on the following text. Use warm, natural, and inspiring language. Do not use figurative language."
              f"Do NOT include your internal thoughts or step-by-step process. Just give the final script:\n\n{text}")
    data = {"model": "tngtech/deepseek-r1t-chimera:free", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.status_code != 200:
            st.error(f"❌ ERROR: {response.status_code} - {response.text}")
            return "[SUMMARY FAILED]"
        
        content = response.json()['choices'][0]['message']['content']
        parts = re.split(r'(?:narasi|narration)\s*:', content, maxsplit=1, flags=re.IGNORECASE)
        
        if len(parts) > 1:
            cleaned_content = parts[1]
        else:
            cleaned_content = content

        match = re.search(r"\*\*(.*?)\*\*", cleaned_content, re.DOTALL)
        if match:
            cleaned_content = match.group(1)

        return cleaned_content.strip()
    except Exception as e:
        st.error(f"❌ ERROR saat memanggil API DeepSeek: {e}")
        return "[SUMMARY FAILED]"

@st.cache_data # Cache hasil terjemahan agar tidak menerjemahkan ulang teks yang sama
def translate_text(text, target_lang='id'):
    """Menerjemahkan teks menggunakan Google Translate."""
    if not text:
        return ""
    try:
        translator = Translator()
        translated = translator.translate(text, dest=target_lang)
        st.success(f"Teks berhasil diterjemahkan ke {target_lang}.")
        return translated.text
    except Exception as e:
        st.error(f"Gagal menerjemahkan teks: {e}")
        return text

def text_to_speech(text, output_audio_path, lang='en'):
    """Mengonversi teks menjadi file audio MP3 dengan bahasa yang dapat dipilih."""
    try:
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        tts = gTTS(text=text, lang=lang)
        tts.save(output_audio_path)
        return output_audio_path
    except Exception as e:
        st.error(f"Gagal membuat file audio: {e}")
        return None

def download_single_image_from_pexels(query, api_key, index=0):
    if not api_key:
        st.error("API Key Pexels tidak ditemukan. Harap atur di file .env atau secrets Hugging Face.")
        return None
    headers = {"Authorization": api_key}
    response = requests.get(f"https://api.pexels.com/v1/search?query={query}&per_page=1", headers=headers)
    data = response.json()
    os.makedirs("temp_images", exist_ok=True)
    if "photos" in data and data["photos"]:
        image_url = data["photos"][0]["src"]["landscape"]
        img_data = requests.get(image_url).content
        image = Image.open(BytesIO(img_data)).resize((1280, 720))
        image_path = f"temp_images/temp_bg_{hashlib.md5(query.encode('utf-8')).hexdigest()}_{index}.jpg"
        image.save(image_path)
        return image_path
    else:
        black_img_path = f"temp_images/temp_bg_black_{index}.jpg"
        Image.new("RGB", (1280, 720), (0, 0, 0)).save(black_img_path)
        return black_img_path

def create_video_with_audio_and_text(audio_path, text, pexels_api_key_local, output_path, progress_callback=None):
    try:
        audio = AudioFileClip(audio_path)
        duration = audio.duration
        chunks = textwrap.wrap(text, width=40)
        num_chunks = len(chunks)
        if num_chunks == 0:
            st.warning("Tidak ada teks untuk ditampilkan di video.")
            return None
        chunk_duration = duration / num_chunks

        video_clips = []
        temp_image_paths = []

        for i, chunk in enumerate(chunks):
            base_query = " ".join(chunk.split()[:5])
            query = f"islamic {base_query}"
            
            bg_path = download_single_image_from_pexels(query, pexels_api_key_local, index=i)
            if bg_path is None: return None
            temp_image_paths.append(bg_path)
            
            text_img = create_text_image_with_pillow(chunk, (1280, 720))
            text_img_path = f"temp_images/text_{i}.png"
            text_img.save(text_img_path)
            temp_image_paths.append(text_img_path)
            
            background_clip = ImageClip(bg_path).set_duration(chunk_duration)
            text_clip = ImageClip(text_img_path).set_duration(chunk_duration).set_position(("center", "center"))
            
            video_clip = CompositeVideoClip([background_clip, text_clip])
            video_clips.append(video_clip)
            if progress_callback:
                progress_callback((i + 1) / num_chunks, f"Membuat klip video {i+1}/{num_chunks}...")

        final_video = concatenate_videoclips(video_clips).set_audio(audio)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if progress_callback: progress_callback(1.0, "Menyelesaikan video...")
            
        final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", logger=None)

        for path in temp_image_paths:
            if os.path.exists(path):
                try: os.remove(path)
                except OSError: pass
        
        return output_path
    except Exception as e:
        st.error(f"Gagal membuat video: {e}")
        return None

# ======================================================================================
# Antarmuka Pengguna (UI) Streamlit
# ======================================================================================

# Inisialisasi session state untuk histori
if 'history' not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR ---
st.sidebar.title("⚙️ Pengaturan & Histori")

language_options = {
    "Indonesia": {"name": "Indonesian", "code": "id"},
    "English": {"name": "English", "code": "en"},
}
selected_language_name = st.sidebar.selectbox(
    "Pilih Bahasa Output:",
    options=list(language_options.keys())
)
selected_language_details = language_options[selected_language_name]

if st.sidebar.button("🗑️ Hapus Histori"):
    for item in st.session_state.history:
        if os.path.exists(item["video_path"]): os.remove(item["video_path"])
        if os.path.exists(item["audio_path"]): os.remove(item["audio_path"])
    st.session_state.history = []
    st.sidebar.success("Histori berhasil dihapus!")
    st.rerun()

st.sidebar.subheader("📜 Histori Video")
if not st.session_state.history:
    st.sidebar.info("Belum ada histori video yang dibuat.")
else:
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.sidebar.expander(f"**{i+1}. {item['question']}**"):
            st.write(f"**Narasi:**")
            st.caption(item['summary'])
            try:
                video_file = open(item['video_path'], 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
            except FileNotFoundError:
                st.error("File video tidak ditemukan.")

# --- HALAMAN UTAMA ---
st.title("🎬 AI Video Generator Otomatis")
st.markdown("Masukkan pertanyaan Anda, dan AI akan membuatkan video singkat lengkap dengan narasi dan gambar latar bernuansa Islami.")

st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50; color: white; border-radius: 12px;
        padding: 10px 24px; font-size: 16px;
    }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

model = load_model()
index, chunk_sources, all_chunks = load_data()

if index and model and all_chunks:
    question = st.text_input("Masukkan pertanyaan Anda di sini:", placeholder="Contoh: Apa kata Islam tentang pentingnya pemuda?")

    if st.button("🚀 Buat Video", type="primary"):
        if not question:
            st.warning("Silakan masukkan pertanyaan terlebih dahulu.")
        elif not openrouter_api_key or not pexels_api_key:
            st.error("Harap pastikan API Key OpenRouter dan Pexels sudah diatur dalam file .env Anda.")
        else:
            progress_placeholder = st.empty()
            results_placeholder = st.empty()
            
            bar = progress_placeholder.progress(0, text="Memulai proses...")
            
            def update_progress(percentage, text):
                bar.progress(int(percentage * 100), text=text)

            update_progress(0.1, "Mencari informasi relevan...")
            results = search_best_chunk(question, model, index, all_chunks, chunk_sources)
            if not results:
                st.error("Tidak dapat menemukan informasi yang relevan dengan pertanyaan Anda."); st.stop()
            source, answer = results[0]

            update_progress(0.25, "Membuat ringkasan narasi dalam Bahasa Inggris (dasar)...")
            english_summary = summarize_with_deepseek(answer, openrouter_api_key, output_language="English")
            if not english_summary or english_summary == "[SUMMARY FAILED]":
                st.error("Gagal membuat ringkasan dasar."); st.stop()

            summarized_answer = english_summary
            if selected_language_details['code'] != 'en':
                update_progress(0.4, f"Menerjemahkan narasi ke Bahasa {selected_language_name}...")
                summarized_answer = translate_text(english_summary, target_lang=selected_language_details['code'])
                if summarized_answer == english_summary:
                    st.warning("Peringatan: Proses terjemahan mungkin gagal, menggunakan teks asli Bahasa Inggris.")
            
            update_progress(0.5, "Mengubah teks menjadi suara...")
            file_hash = hashlib.md5(summarized_answer.encode('utf-8')).hexdigest()
            output_dir = "output"
            audio_path = os.path.join(output_dir, f"audio_{file_hash}.mp3")
            video_path = os.path.join(output_dir, f"video_{file_hash}.mp4")
            
            audio_path = text_to_speech(summarized_answer, audio_path, lang=selected_language_details['code'])
            if not audio_path:
                st.error("Gagal membuat file audio."); st.stop()

            def video_progress_callback(p, t):
                scaled_progress = 0.6 + p * 0.35
                update_progress(scaled_progress, t)

            video_path = create_video_with_audio_and_text(audio_path, summarized_answer, pexels_api_key, video_path, progress_callback=video_progress_callback)
            
            update_progress(1.0, "Proses Selesai!")
            progress_placeholder.empty()

            if video_path and os.path.exists(video_path):
                with results_placeholder.container():
                    st.success("✅ Video berhasil dibuat!")
                    st.subheader("Narasi Video:")
                    st.write(summarized_answer)
                    st.write(f"*(Sumber informasi: `{source}`)*")

                    with open(video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)
                    st.download_button(
                        label="📥 Unduh Video (MP4)", data=video_bytes,
                        file_name=f"generated_video_{file_hash[:8]}.mp4", mime="video/mp4"
                    )
                
                st.session_state.history.append({
                    "question": question,
                    "summary": summarized_answer,
                    "source": source,
                    "video_path": video_path,
                    "audio_path": audio_path,
                })
                st.rerun() 
            else:
                results_placeholder.error("Terjadi kesalahan dan video tidak dapat dibuat.")
else:
    st.error("Aplikasi tidak dapat dimulai karena gagal memuat data atau model. Mohon periksa kembali pesan error di atas.")