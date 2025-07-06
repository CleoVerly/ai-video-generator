# app.py - Versi Final Lengkap dan Teruji

import streamlit as st
import os
import requests
from dotenv import load_dotenv
import time
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Import library untuk video
from gtts import gTTS
from moviepy.editor import ImageClip, CompositeVideoClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Import library untuk terjemahan yang kompatibel
from deep_translator import GoogleTranslator

# --- 1. KONFIGURASI DAN INISIALISASI ---

load_dotenv()
st.set_page_config(layout="wide", page_title="AI Video Generator")

# Konfigurasi Path (gunakan os.path.join untuk kompatibilitas)
DATA_DIR = "data"
FAISS_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
MAPPING_PATH = os.path.join(DATA_DIR, "chunk_mapping.json")
CHUNKS_PATH = os.path.join(DATA_DIR, "epub_chunks_translated.json")

MODEL_NAME_EMBEDDING = 'sentence-transformers/all-MiniLM-L6-v2'
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
VIDEO_OUTPUT_DIR = "generated_videos"
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)


# --- 2. FUNGSI-FUNGSI HELPER ---

def load_tailwind_cdn():
    """Menyuntikkan skrip Tailwind CSS."""
    st.markdown("""
        <script src="https://cdn.tailwindcss.com"></script>
        <style> #MainMenu, footer { visibility: hidden; } </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    """Memuat model embedding dan data dari FAISS & JSON."""
    print("Memuat model embedding dan data...")
    model = SentenceTransformer(MODEL_NAME_EMBEDDING, device='cpu')

    # Cek keberadaan semua file data
    if not all(os.path.exists(p) for p in [FAISS_PATH, MAPPING_PATH, CHUNKS_PATH]):
        st.error(f"Satu atau lebih file data tidak ditemukan. Pastikan path sudah benar: {FAISS_PATH}, {MAPPING_PATH}, {CHUNKS_PATH}")
        return None, None, None

    index = faiss.read_index(FAISS_PATH)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("Model dan data berhasil dimuat.")
    return model, index, chunks

def search_similar_chunks(query, _model, _index, _chunks, k=5):
    """Mencari chunk yang relevan menggunakan FAISS dengan penanganan error yang lebih baik."""
    query_embedding = _model.encode([query])
    distances, indices = _index.search(query_embedding.astype('float32'), k=k)

    relevant_chunks = []
    for i in indices[0]:
        chunk_key = str(i)
        if chunk_key in _chunks:
            relevant_chunks.append(_chunks[chunk_key])
        else:
            print(f"Peringatan: Indeks FAISS {i} tidak ditemukan dalam data chunk.")

    if not relevant_chunks:
        return "Maaf, tidak ditemukan konteks yang relevan untuk pertanyaan ini."

    return "\n\n---\n\n".join(relevant_chunks)


def get_answer_from_ai(question, context):
    """Mendapatkan jawaban dari AI melalui OpenRouter."""
    prompt = f"Berdasarkan konteks berikut:\n\n{context}\n\nJawablah pertanyaan ini dalam 1-2 kalimat singkat dan jelas dalam Bahasa Indonesia: '{question}'"

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={
            "model": "deepseek/deepseek-r1-0528:free",
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=45
    )
    response.raise_for_status()
    answer = response.json()['choices'][0]['message']['content']
    print("Jawaban diterima dari OpenRouter.")
    return answer

def translate_text(text, dest_lang='id'):
    """Menerjemahkan teks ke bahasa tujuan menggunakan deep-translator."""
    try:
        # Format bahasa untuk deep-translator adalah 'indonesian', bukan 'id'
        lang_map = {'id': 'indonesian'}
        target_lang = lang_map.get(dest_lang, dest_lang)
        
        # Inisialisasi dan terjemahkan
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated
    except Exception as e:
        print(f"Error saat menerjemahkan: {e}")
        return text # Kembalikan teks asli jika gagal

def create_text_image_with_pillow(text, size=(1280, 720), font_size=50):
    """Membuat gambar PNG transparan yang berisi teks menggunakan Pillow."""
    image = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        print("Font Arial tidak ditemukan, menggunakan font default.")
        font = ImageFont.load_default(size=font_size)

    wrapped_text = "\n".join(textwrap.wrap(text, width=40))
    text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
    position = ((size[0] - (text_bbox[2] - text_bbox[0])) / 2, (size[1] - (text_bbox[3] - text_bbox[1])) / 2)

    stroke_width = 2
    x, y = position
    draw.multiline_text((x-stroke_width, y), wrapped_text, font=font, fill=(0,0,0,255))
    draw.multiline_text((x+stroke_width, y), wrapped_text, font=font, fill=(0,0,0,255))
    draw.multiline_text((x, y-stroke_width), wrapped_text, font=font, fill=(0,0,0,255))
    draw.multiline_text((x, y+stroke_width), wrapped_text, font=font, fill=(0,0,0,255))
    draw.multiline_text(position, wrapped_text, font=font, fill=(255,255,255,255))

    text_image_path = os.path.join(VIDEO_OUTPUT_DIR, "temp_text.png")
    image.save(text_image_path)
    return text_image_path

def create_video(audio_path, text_answer, bg_image_path):
    """Membuat video dengan teks yang dirender oleh Pillow."""
    audio_clip = AudioFileClip(audio_path)
    background_clip = ImageClip(bg_image_path, duration=audio_clip.duration)
    text_image_path = create_text_image_with_pillow(text_answer, size=background_clip.size)
    text_clip = ImageClip(text_image_path, duration=audio_clip.duration)

    final_clip = CompositeVideoClip([background_clip, text_clip]).set_audio(audio_clip)
    video_filename = f"output_{int(time.time())}.mp4"
    video_output_path = os.path.join(VIDEO_OUTPUT_DIR, video_filename)
    final_clip.write_videofile(video_output_path, fps=24, codec='libx264', audio_codec='aac', logger=None)

    os.remove(audio_path)
    os.remove(text_image_path)
    if os.path.basename(bg_image_path) == "background.jpg":
        os.remove(bg_image_path)
        
    return video_output_path

def search_pexels_image(query):
    """Mencari dan men-download gambar dari Pexels."""
    search_query = re.sub(r'\W+', ' ', query).strip()
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": search_query or "abstract technology", "per_page": 1, "orientation": "landscape"}
    try:
        response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params)
        response.raise_for_status()
        if response.json().get('photos'):
            image_url = response.json()['photos'][0]['src']['large']
            image_path = os.path.join(VIDEO_OUTPUT_DIR, "background.jpg")
            with open(image_path, 'wb') as f: f.write(requests.get(image_url).content)
            Image.open(image_path).resize((1280, 720)).save(image_path)
            return image_path
    except Exception as e:
        print(f"Gagal mengambil gambar dari Pexels: {e}")

    fallback_path = os.path.join(VIDEO_OUTPUT_DIR, "background.jpg")
    Image.new('RGB', (1280, 720), color='#1a202c').save(fallback_path)
    return fallback_path


# --- 3. MEMBANGUN APLIKASI STREAMLIT ---

load_tailwind_cdn()
embedding_model, faiss_index, all_chunks = load_models_and_data()

st.markdown("<h1 class='text-4xl font-bold text-center text-gray-800 my-4'>🎬 AI Video Generator</h1>", unsafe_allow_html=True)

if embedding_model and faiss_index and all_chunks:
    prompt = st.text_input("Masukkan topik atau pertanyaan untuk video Anda:", key="prompt_input")
    
    if st.button("Buat Video", use_container_width=True, type="primary"):
        if prompt:
            try:
                with st.spinner("Mencari konteks yang relevan..."):
                    context = search_similar_chunks(prompt, embedding_model, faiss_index, all_chunks)
                
                with st.spinner("Meminta jawaban dari AI..."):
                    answer_text = get_answer_from_ai(prompt, context)
                st.info(f"**Jawaban AI:** {answer_text}")

                with st.spinner("Membuat Audio dari Jawaban..."):
                    # Karena AI sudah menjawab dalam Bahasa Indonesia, kita tidak perlu menerjemahkan lagi.
                    audio_tts = gTTS(text=answer_text, lang='id', slow=False)
                    audio_path = os.path.join(VIDEO_OUTPUT_DIR, f"speech.mp3")
                    audio_tts.save(audio_path)

                with st.spinner("Mencari gambar latar..."):
                    bg_image_path = search_pexels_image(query=prompt)

                with st.spinner("Merender video... Proses ini mungkin butuh waktu."):
                    final_video_path = create_video(audio_path, answer_text, bg_image_path)
                    
                st.success("Video berhasil dibuat!")
                st.video(final_video_path)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
        else:
            st.warning("Silakan masukkan topik atau pertanyaan.")