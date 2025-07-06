# app.py - Versi Final Disesuaikan dengan Struktur Data Anda

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
import nltk

# Import library untuk video
from gtts import gTTS
from moviepy.editor import ImageClip, CompositeVideoClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Import library untuk terjemahan
from deep_translator import GoogleTranslator

# Download 'punkt' untuk pemisahan kalimat
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- 1. KONFIGURASI DAN INISIALISASI ---
load_dotenv()
st.set_page_config(layout="wide", page_title="AI Video Generator")

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
    st.markdown("""
        <script src="https://cdn.tailwindcss.com"></script>
        <style> #MainMenu, footer { visibility: hidden; } </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    model = SentenceTransformer(MODEL_NAME_EMBEDDING, device='cpu')
    if not all(os.path.exists(p) for p in [FAISS_PATH, MAPPING_PATH, CHUNKS_PATH]):
        st.error("Satu atau lebih file data (FAISS/JSON) tidak ditemukan.")
        return None, None, None, None
    index = faiss.read_index(FAISS_PATH)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f: mapping = json.load(f)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f: chunks = json.load(f)
    return model, index, chunks, mapping

# ### FUNGSI INI TELAH DIPERBAIKI SESUAI FORMAT DATA ANDA ###
def search_similar_chunks(query, _model, _index, _chunks, _mapping, k=5):
    """Mencari chunk relevan dan mengekstrak sumber dari daftar mapping."""
    query_embedding = _model.encode([query])
    distances, indices = _index.search(query_embedding.astype('float32'), k=k)
    
    relevant_chunks = []
    source_files = set() 
    for i in indices[0]:
        # Pastikan indeks tidak melebihi panjang list/dictionary
        if i < len(_mapping) and str(i) in _chunks:
            # Ambil teks chunk
            relevant_chunks.append(_chunks[str(i)])
            
            # Ambil deskripsi dari mapping dan ekstrak nama kitab
            description = _mapping[i]
            # Pisahkan string berdasarkan " - Chunk " dan ambil bagian pertama
            source_name = description.split(" - Chunk ")[0]
            source_files.add(source_name.strip())
    
    if not relevant_chunks:
        return "Tidak ada konteks relevan.", []
            
    return "\n\n---\n\n".join(relevant_chunks), list(source_files)

# Sisa fungsi helper lainnya tetap sama
def get_answer_from_ai(question, context):
    prompt = f"Berdasarkan konteks berikut:\n\n{context}\n\nJawablah pertanyaan ini dengan jelas dalam Bahasa Indonesia: '{question}'"
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={"model": "deepseek/deepseek-r1-0528:free", "messages": [{"role": "user", "content": prompt}]},
        timeout=60
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def translate_text(text, dest_lang='id'):
    try:
        return GoogleTranslator(source='auto', target=dest_lang).translate(text)
    except Exception as e:
        print(f"Error saat menerjemahkan: {e}")
        return text

def search_pexels_images(query, count=5):
    search_query = re.sub(r'\W+', ' ', query).strip() or "nature"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": search_query, "per_page": count, "orientation": "landscape"}
    image_paths = []
    try:
        response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params)
        response.raise_for_status()
        photos = response.json().get('photos', [])
        for i, photo in enumerate(photos):
            image_url = photo['src']['large']
            image_path = os.path.join(VIDEO_OUTPUT_DIR, f"background_{i}.jpg")
            with open(image_path, 'wb') as f: f.write(requests.get(image_url).content)
            Image.open(image_path).resize((1280, 720)).save(image_path)
            image_paths.append(image_path)
    except Exception as e:
        print(f"Gagal mengambil gambar dari Pexels: {e}")
    while len(image_paths) < count:
        fallback_path = os.path.join(VIDEO_OUTPUT_DIR, f"background_{len(image_paths)}.jpg")
        Image.new('RGB', (1280, 720), color='#1a202c').save(fallback_path)
        image_paths.append(fallback_path)
    return image_paths

def create_text_image_with_pillow(text, size=(1280, 720), font_size=50):
    image = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
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
    text_image_path = os.path.join(VIDEO_OUTPUT_DIR, f"temp_text_{int(time.time()*1000)}.png")
    image.save(text_image_path)
    return text_image_path

def create_dynamic_video(full_text, bg_image_paths):
    sentences = nltk.sent_tokenize(full_text)
    video_clips = []
    files_to_delete = []
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip(): continue
        audio_filename = f"speech_{i}.mp3"
        audio_path = os.path.join(VIDEO_OUTPUT_DIR, audio_filename)
        tts = gTTS(text=sentence, lang='id', slow=False)
        tts.save(audio_path)
        files_to_delete.append(audio_path)
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        bg_path = bg_image_paths[i % len(bg_image_paths)]
        background_clip = ImageClip(bg_path, duration=duration)
        text_image_path = create_text_image_with_pillow(sentence, size=background_clip.size)
        files_to_delete.append(text_image_path)
        text_clip = ImageClip(text_image_path, duration=duration)
        segment_clip = CompositeVideoClip([background_clip, text_clip]).set_audio(audio_clip)
        video_clips.append(segment_clip)

    if not video_clips:
        raise ValueError("Tidak ada segmen video yang bisa dibuat.")

    final_clip = concatenate_videoclips(video_clips)
    video_filename = f"output_{int(time.time())}.mp4"
    video_output_path = os.path.join(VIDEO_OUTPUT_DIR, video_filename)
    final_clip.write_videofile(video_output_path, fps=24, codec='libx264', audio_codec='aac', logger=None)
    
    for f in files_to_delete + bg_image_paths:
        if os.path.exists(f): os.remove(f)
        
    return video_output_path


# --- 3. MEMBANGUN APLIKASI STREAMLIT ---

load_tailwind_cdn()
embedding_model, faiss_index, all_chunks, mapping = load_models_and_data()

st.markdown("<h1 class='text-4xl font-bold text-center text-gray-800 my-4'>🎬 AI Video Generator Dinamis</h1>", unsafe_allow_html=True)

if embedding_model and faiss_index and all_chunks:
    prompt = st.text_input("Masukkan topik atau pertanyaan untuk video Anda:", key="prompt_input")
    
    if st.button("Buat Video", use_container_width=True, type="primary"):
        if prompt:
            try:
                with st.spinner("Mencari konteks dan sumber..."):
                    context, sources = search_similar_chunks(prompt, embedding_model, faiss_index, all_chunks, mapping)
                
                with st.spinner("Meminta jawaban dari AI..."):
                    answer_text = get_answer_from_ai(prompt, context)
                
                if sources:
                    source_str = ", ".join(sources)
                    st.caption(f"ℹ️ Jawaban diambil berdasarkan konteks dari: **{source_str}**")
                
                st.info(f"**Narasi Video:** {answer_text}")

                with st.spinner("Menerjemahkan & Membuat Audio..."):
                    translated_text = translate_text(answer_text, dest_lang='id')
                    
                with st.spinner("Mencari gambar latar belakang..."):
                    bg_image_paths = search_pexels_images(query=prompt, count=5)

                with st.spinner("Merender video... Proses ini paling lama."):
                    final_video_path = create_dynamic_video(translated_text, bg_image_paths)
                    
                st.success("Video berhasil dibuat!")
                st.video(final_video_path)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
        else:
            st.warning("Silakan masukkan topik atau pertanyaan.")