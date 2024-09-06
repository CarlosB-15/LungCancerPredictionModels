import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------- Percorsi delle cartelle 
train_dir = 'c:\\Users\\carlo\\Desktop\\TESI\\codice tesi\\archive-image - PREFERITO\Data\\train'
valid_dir = 'c:\\Users\\carlo\\Desktop\\TESI\\codice tesi\\archive-image - PREFERITO\Data\\valid'
test_dir = 'c:\\Users\\carlo\\Desktop\\TESI\\codice tesi\\archive-image - PREFERITO\Data\\test'

# Parametri del modello
img_width, img_height = 224, 224  # Dimensione delle immagini in cui verranno ridimensionate
batch_size = 32                   # indica il numero di immagini elaborate insieme in un'unica iterazione di addestramento.
epochs = 20                       # rappresenta il numero di iterazioni complete sul set di addestramento
num_classes = 4                   # 3 tipologie di cancro + 1 normale

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

# Per la validazione e il test, le immagini sono semplicemente scalate (normalizzate) a valori compresi tra 0 e 1
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# caricano le immagini dalle directory specificate e le preparano per l'addestramento, validazione e test rispettivamente
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical' # indica che il problema è una classificazione multiclasse (con più di due classi), quindi le etichette verranno codificate come vettori one-hot.
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

#-------------------------------------------- Creazione del modello

# Il modello è una CNN costituita da:
# - Tre strati convoluzionali (Conv2D) con 32, 64 e 128 filtri rispettivamente, ognuno seguito da uno strato di pooling (MaxPooling2D) che riduce la dimensionalità delle mappe di 
# attivazione.
# - Uno strato fully connected (Dense) con 128 neuroni e funzione di attivazione ReLU.
# - Uno strato di dropout (Dropout) che disattiva casualmente il 50% dei neuroni durante l'addestramento per prevenire l'overfitting.
# - Uno strato di uscita (Dense) con un numero di neuroni pari al numero di classi (num_classes), con funzione di attivazione softmax per produrre una probabilità per ciascuna 
# classe.

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
 
#--------------------------------------------Compilazione del modello
# Il modello viene compilato utilizzando l'ottimizzatore Adam con un learning rate di 0.0001, la funzione di perdita categorical_crossentropy (adeguata per la classificazione 
# multiclasse) e la metrica di accuratezza (accuracy).

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#--------------------------------------------Addestramento del modello
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=valid_generator
)


#-------------------------------------------- PER USARE IL MODELLO CREATO SENZA DOVERLO ADDESTRARE NUOVAMENTE 
#                                             (commentare Creazione del modello, Compilazione del modello e Addestramento del modello)
# model = load_model('cancer_classification_model.h5')

# Valutazione del modello sul set di test
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.2f}')

#-------------------------------------------- Salvataggio del modello
# Il modello addestrato viene salvato su disco in un file con estensione .h5 (cancer_classification_model.h5), permettendo di ricaricarlo e riutilizzarlo senza doverlo riaddestrare.

model.save('cancer_classification_model.h5')



# Valutazione del modello sul set di test
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.2f}')

# Ottieni le etichette vere dal set di test
test_labels = test_generator.classes

# Ottieni le previsioni per il set di test
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Genera la matrice di confusione
cm = confusion_matrix(test_labels, predicted_classes)

# Visualizza la matrice di confusione
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title('Matrice di Confusione')
plt.show()



# Funzione per fare previsioni su nuove immagini
def predict_image(image_path):
    from tensorflow.keras.preprocessing import image
    import numpy as np
    
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    predicted_class = class_indices[np.argmax(prediction)]
    
    return predicted_class

# Esempio di utilizzo della funzione di previsione
image_path = 'c:\\Users\\carlo\\Desktop\\TESI\\codice tesi\\000058 (5)  (per fare uscire squamos cell).png'
predicted_class = predict_image(image_path)
print(f'\nThe predicted class is: {predicted_class}')