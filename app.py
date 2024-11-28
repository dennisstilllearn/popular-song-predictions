import streamlit as st
import numpy as np
import pickle

# Memuat Pipeline dan y Transformer
with open('modelxscaler_pipeline.pkl', 'rb') as file:
    loaded_objects = pickle.load(file)

loaded_pipeline = loaded_objects['pipeline']
loaded_y_transformer = loaded_objects['y_transformer']

# Judul dan Deskripsi
st.title("🎵 Prediksi Popularitas Lagu")
st.markdown("""
Masukkan detail lagu di bawah ini untuk memprediksi popularitasnya. 🚀
""")

st.markdown("---")

# Input Judul Lagu (untuk tampilan)
st.header("🎵 Masukkan Judul Lagu")
song_title = st.text_input("Masukkan Judul Lagu Yang Ingin Anda Prediksi")

# Membagi input menjadi beberapa bagian untuk tampilan lebih rapi
st.header("🔧 Masukkan Detail Lagu")
col1, col2 = st.columns(2)

with col1:
    song_duration_ms = st.number_input("Duration (in ms)", 0, 1000000)
    acousticness = st.number_input("Acousticness", 0.0, 1.0, format="%.6f")
    danceability = st.number_input("Danceability", 0.0, 1.0, format="%.6f")
    energy = st.number_input("Energy", 0.0, 1.0, format="%.6f")
    instrumentalness = st.number_input("Instrumentalness", 0.0, 1.0, format="%.6f")
    key = st.selectbox("Key (Basic Note)", range(0, 12))  # 0=C, 1=C#/Db, 2=D, ...

with col2:
    liveness = st.number_input("Liveness", 0.0, 1.0, format="%.6f")
    loudness = st.number_input("Loudness (in dB)", -60.0, 0.0, format="%.6f")
    audio_mode = st.selectbox("Audio Mode", [0, 1])
    speechiness = st.number_input("Speechiness", 0.0, 1.0, format="%.6f")
    tempo = st.number_input("Tempo (BPM)", min_value=0.0, max_value=300.0, step=0.001, format="%.3f")
    time_signature = st.selectbox("Time Signature", [3, 4, 5])
    audio_valence = st.number_input("Valence", 0.0, 1.0, format="%.6f")



# Menggabungkan input data ke dalam array
data = np.array([[song_duration_ms, acousticness, danceability, energy, instrumentalness, key, 
                  liveness, loudness, audio_mode, speechiness, tempo, time_signature, audio_valence]])

# Tombol prediksi
if st.button("Prediksi Popularitas Lagu"):
    st.subheader("Hasil Prediksi")
    try:
        # Melakukan prediksi
        y_pred_transformed = loaded_pipeline.predict(data)  # Pipeline otomatis memproses input
        y_pred_actual = loaded_y_transformer.inverse_transform(y_pred_transformed.reshape(-1, 1))  # Membalik transformasi output
        
        if song_title:
            st.success(f"Lagu '{song_title}' memiliki tingkat popularitas sebesar **{y_pred_actual[0][0]:.2f}**")
        else:
            st.success(f"Lagu ini memiliki tingkat popularitas sebesar **{y_pred_actual[0][0]:.2f}**")
        
        st.balloons()
    except Exception as e:
        st.error(f"Terjadi kesalahan selama prediksi: {e}")

# Footer
st.markdown("---")
footer_text = """
    Irfansyah Nur Aviatna | Dennis Raka Pratama Mukti | Muhammad Ilham | Novian Rizky Dewanta | Khafidhotul Urfa | Aulia Husna
"""

# Menampilkan footer dengan spasi untuk memastikan berada di bawah
st.markdown(footer_text, unsafe_allow_html=False)
