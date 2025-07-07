# app.py
import streamlit as st
import pandas as pd
import pickle

# Model ve yardımcılar
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("selector.pkl", "rb") as f:
    selector = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Streamlit başlık
st.set_page_config(page_title="📱 Mobil Fiyat Tahmini", page_icon="📲", layout="centered")
st.title("📱 Mobil Fiyat Aralığı Tahmin Sistemi")

# Seçilen özellik isimlerini al
selected_indices = selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_indices]

st.markdown("Aşağıya telefonun özelliklerini girin:")

# Girdi formu
input_data = {}
for feature in selected_features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("Tahmin Et"):
    # Tüm sütunlar sıfırla başlatılır, sadece seçilenler girilir
    full_input = {f: 0.0 for f in feature_names}
    for f in selected_features:
        full_input[f] = input_data[f]

    user_df = pd.DataFrame([full_input])
    user_scaled = scaler.transform(user_df)
    user_selected = selector.transform(user_scaled)

    prediction = model.predict(user_selected)[0]

    st.subheader("📊 Tahmin Sonucu")
    st.info(f"Bu telefonun fiyat aralığı: **{prediction}**")
    st.caption("""
    0 → Düşük  
    1 → Orta  
    2 → Yüksek  
    3 → Çok Yüksek  
    """)
