import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------------------- Percorsi delle cartelle 
train_dir = 'c:\\Users\\carlo\\Desktop\\TESI\\codice tesi\\archive-image - PREFERITO\Data\\train'

# Parametri del modello
img_width, img_height = 224, 224  # Dimensione delle immagini in cui verranno ridimensionate
batch_size = 32                   # indica il numero di immagini elaborate insieme in un'unica iterazione di addestramento.

# -------------------------------------------- Preprocessing delle immagini

# ImageDataGenerator è utilizzato per applicare delle trasformazioni (augmentation) alle immagini di addestramento, come rotazioni, 
# traslazioni e zoom, per migliorare la robustezza del modello. 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# caricano le immagini dalle directory specificate e le preparano per l'addestramento, validazione e test rispettivamente
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical' # indica che il problema è una classificazione multiclasse (con più di due classi), quindi le etichette verranno codificate come vettori one-hot.
)

model = load_model('cancer_classification_model.h5')

# Funzione per fare previsioni su nuove immagini
def predict_image(img):

    # Verifica se l'immagine è in formato RGBA e converti in RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')
        
    img = img.resize((img_width, img_height))  # Ridimensiona l'immagine a 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    predicted_class = class_indices[np.argmax(prediction)]
    
    return predicted_class

# Interfaccia Streamlit
st.title("Diagnosi del carcinoma polmonare mediante utilizzo di immagini TC")

st.write("Carica la tua TC polmonare per ottenere una diagnosi in pochi secondi!")

# Caricamento dell'immagine
uploaded_file = st.file_uploader("Seleziona un file immagine (formati supportati: jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostra l'immagine caricata
    img = Image.open(uploaded_file)
    st.image(img, caption='Immagine Caricata con Successo!', use_column_width=True)
    
    # Prevedi la classe
    predicted_class = predict_image(img)
    
    # Mostra la classe prevista
    st.markdown(f"# Risultato della Diagnosi: **{predicted_class}**")

