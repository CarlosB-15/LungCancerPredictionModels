import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Caricamento del dataset
df = pd.read_csv('c:/Users/carlo/Desktop/TESI/codice tesi/cancer patient data sets.csv')

# # Preprocessing
df = df.drop(columns=['index', 'Patient Id'])

# # Codifica delle etichette per la colonna target 'Level'
label_encoder = LabelEncoder()
df['Level'] = label_encoder.fit_transform(df['Level'])

# Titolo dell'applicazione
st.title("Valutazione del Rischio di Cancro ai Polmoni")

# Input da parte dell'utente
age = st.number_input("Inserisci la tua età", min_value=0, max_value=120, value=25)
gender = st.selectbox("Seleziona il tuo genere", options=[1, 2], format_func=lambda x: "Maschio" if x == 1 else "Femmina")
air_pollution = st.slider("Inquinamento dell'aria", min_value=1, max_value=10, value=5)
alcohol_use = st.slider("Uso di alcol", min_value=1, max_value=10, value=5)
dust_allergy = st.slider("Allergia alla polvere", min_value=1, max_value=10, value=5)
occupational_hazards = st.slider("Rischi occupazionali", min_value=1, max_value=10, value=5)
genetic_risk = st.slider("Rischio genetico", min_value=1, max_value=10, value=5)
chronic_lung_disease = st.slider("Malattia polmonare cronica", min_value=1, max_value=10, value=5)
balanced_diet = st.slider("Dieta bilanciata", min_value=1, max_value=10, value=5)
obesity = st.slider("Obesità", min_value=1, max_value=10, value=5)
smoking = st.slider("Fumo", min_value=1, max_value=10, value=5)
passive_smoker = st.slider("Fumatore passivo", min_value=1, max_value=10, value=5)
chest_pain = st.slider("Dolore al petto", min_value=1, max_value=10, value=5)
coughing_of_blood = st.slider("Tosse con sangue", min_value=1, max_value=10, value=5)
fatigue = st.slider("Fatica", min_value=1, max_value=10, value=5)
weight_loss = st.slider("Perdita di peso", min_value=1, max_value=10, value=5)
shortness_of_breath = st.slider("Respiro corto", min_value=1, max_value=10, value=5)
wheezing = st.slider("Sibilo", min_value=1, max_value=10, value=5)
swallowing_difficulty = st.slider("Difficoltà di deglutizione", min_value=1, max_value=10, value=5)
clubbing_of_finger_nails = st.slider("Ippocratismo digitale", min_value=1, max_value=10, value=5)
frequent_cold = st.slider("Raffreddore frequente", min_value=1, max_value=10, value=5)
dry_cough = st.slider("Tosse secca", min_value=1, max_value=10, value=5)
snoring = st.slider("Russamento", min_value=1, max_value=10, value=5)

# Conversione degli input in un array numpy per fare la previsione
input_data = np.array([[age, gender, air_pollution, alcohol_use,
                        dust_allergy, occupational_hazards, genetic_risk, 
                        chronic_lung_disease, balanced_diet, obesity, smoking, passive_smoker, 
                        chest_pain, coughing_of_blood, fatigue, weight_loss,
                        shortness_of_breath, wheezing, swallowing_difficulty, 
                        clubbing_of_finger_nails, frequent_cold, dry_cough, snoring]])

loaded_model = joblib.load('model_random_forest.pkl')

# Bottone per fare la previsione
if st.button("Calcola il Rischio"):
    prediction = loaded_model.predict(input_data)
    result = label_encoder.inverse_transform(prediction)[0]
    
    if result == 'Low':
        st.markdown(f"<h2 style='color: green;'>La possibilità di avere il cancro ai polmoni è: {result}</h2>", unsafe_allow_html=True)
    elif result == 'Medium':
        st.markdown(f"<h2 style='color: orange;'>La possibilità di avere il cancro ai polmoni è: {result}</h2>", unsafe_allow_html=True)
    elif result == 'High':
        st.markdown(f"<h2 style='color: red;'>La possibilità di avere il cancro ai polmoni è: {result}</h2>", unsafe_allow_html=True)

