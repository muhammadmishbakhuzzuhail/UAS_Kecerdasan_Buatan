import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch.nn as nn

st.set_page_config(page_title="Dasbor Analisis Berita", page_icon="🚀", layout="wide")

# --- Fungsi Pemuatan Model (Menggunakan Cache untuk Efisiensi) ---

@st.cache_resource
def load_fakenews_model():
    """Memuat model dan tokenizer untuk deteksi berita palsu."""
    st.info("Memuat model Deteksi Berita Palsu...")
    tokenizer = AutoTokenizer.from_pretrained("vikram71198/distilroberta-base-finetuned-fake-news-detection")
    model = AutoModelForSequenceClassification.from_pretrained("vikram71198/distilroberta-base-finetuned-fake-news-detection")
    return tokenizer, model

@st.cache_resource
def load_topic_classifier():
    """Memuat pipeline untuk klasifikasi topik berita."""
    st.info("Memuat model Klasifikasi Topik...")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model="classla/multilingual-IPTC-news-topic-classifier", device=device, max_length=512, truncation=True)
    return classifier

@st.cache_resource
def load_summarizer():
    """Memuat pipeline untuk peringkasan teks."""
    st.info("Memuat model Peringkas Teks...")
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model="Falconsai/text_summarization", device=device)
    return summarizer

# --- Memuat semua model di awal ---
# Menampilkan pesan loading saat model diunduh/dimuat untuk pertama kali
with st.spinner("Mempersiapkan semua model AI... Ini mungkin memakan waktu beberapa saat pada pemuatan pertama."):
    fakenews_tokenizer, fakenews_model = load_fakenews_model()
    topic_classifier = load_topic_classifier()
    summarizer = load_summarizer()

# --- Antarmuka Pengguna (UI) Streamlit ---
st.title("🚀 Dasbor Analisis Berita Cerdas")
st.markdown("Analisis berita secara komprehensif: deteksi keaslian, identifikasi topik, dan dapatkan ringkasan instan.")

st.markdown("---")

user_input = st.text_area("Masukkan teks artikel berita yang ingin Anda analisis:", height=250, placeholder="Salin dan tempel artikel berita lengkap di sini...")

analyze_button = st.button("✨ Analisis Sekarang!", type="primary")

# --- Logika Backend dan Tampilan Hasil ---
if analyze_button and user_input:
    # Memastikan input tidak terlalu pendek
    if len(user_input.split()) < 40:
        st.warning("Teks terlalu pendek. Harap masukkan artikel yang lebih panjang untuk hasil yang akurat.")
    else:
        with st.spinner("Menganalisis keaslian, topik, dan membuat ringkasan..."):
            
            # --- Proses 1: Deteksi Berita Palsu ---
            encoded_input = fakenews_tokenizer(user_input, truncation=True, padding="max_length", max_length=512, return_tensors='pt')
            output_logits = fakenews_model(**encoded_input)["logits"]
            softmax = nn.Softmax(dim=1)
            probs = softmax(output_logits.detach())
            prob_real, prob_fake = probs.squeeze().tolist()
            
            jenis_berita_label = "Berita Nyata" if prob_real > prob_fake else "Berita Palsu"
            jenis_berita_score = prob_real if prob_real > prob_fake else prob_fake

            # --- Proses 2: Klasifikasi Topik ---
            topic_result = topic_classifier(user_input)[0]
            tema_label = topic_result['label'].title()
            tema_score = topic_result['score']

            # --- Proses 3: Peringkasan Teks ---
            try:
                # 1. Hitung jumlah token pada teks input.
                input_token_count = len(fakenews_tokenizer.encode(user_input))

                # 2. Tentukan target panjang ringkasan (misal, 20% dari input)
                #    dan batasi (clamp) dalam rentang yang aman (minimal 70, maksimal 250 token).
                target_length = input_token_count // 5  # Target 20% dari panjang input
                
                # Pastikan target tidak kurang dari 70 dan tidak lebih dari 250
                safe_max_length = max(70, min(250, target_length))
                safe_min_length = max(50, safe_max_length // 2) # min_length = setengah dari max_length

                st.info(f"Panjang Input: {input_token_count} token. Target ringkasan: ~{safe_max_length} token.")

                # 3. Panggil summarizer dengan parameter yang sudah dihitung dan aman
                summary_result = summarizer(
                    user_input,
                    max_length=safe_max_length,
                    min_length=safe_min_length,
                    do_sample=False
                )[0]
                ringkasan_teks = summary_result['summary_text']

            except Exception as e:
                st.error(f"Gagal membuat ringkasan: {e}")
                ringkasan_teks = "Model tidak dapat memproses teks ini untuk diringkas."

            # --- Menampilkan Hasil Sesuai Format yang Diminta ---
            st.markdown("---")
            st.header("Hasil Analisis Komprehensif")

            # Baris 1: Jenis Berita
            st.markdown("#### 1. Jenis Berita")
            if jenis_berita_label == "Berita Nyata":
                st.success(f"**{jenis_berita_label}**")
            else:
                st.error(f"**{jenis_berita_label}**")
            
            # Baris 2: Tema
            st.markdown("#### 2. Tema")
            st.info(f"**{tema_label}**")

            # Baris 3: Ringkasan
            st.markdown("#### 3. Ringkasan")
            st.markdown(f"> {ringkasan_teks}")
            
            # Bagian Akurasi di Bawah
            st.markdown("---")
            st.subheader("📊 Tingkat Keyakinan Model")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Keaslian**")
                st.progress(jenis_berita_score)
                st.write(f"{jenis_berita_score:.2%}")
            
            with col2:
                st.markdown("**Topik**")
                st.progress(tema_score)
                st.write(f"{tema_score:.2%}")
            
            with col3:
                st.markdown("**Ringkasan**")
                st.info("Tidak Berlaku (Model Generatif)")


elif analyze_button and not user_input:
    st.error("Mohon masukkan teks berita terlebih dahulu untuk dianalisis.")