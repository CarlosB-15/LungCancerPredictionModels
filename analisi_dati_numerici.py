import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Caricamento del dataset
df = pd.read_csv('c:/Users/carlo/Desktop/TESI/codice tesi/cancer patient data sets.csv')

# Preprocessing
# Rimuovere colonne non necessarie
df = df.drop(columns=['index', 'Patient Id'])

# Codifica delle etichette per la colonna target 'Level'
label_encoder = LabelEncoder()
df['Level'] = label_encoder.fit_transform(df['Level'])

# Separazione delle features e del target
X = df.drop(columns=['Level'])
y = df['Level']

# Suddivisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creazione e addestramento del modello
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Previsione sui dati di test
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

#salvataggio modello
joblib.dump(model, 'model_random_forest.pkl')

# Caricare il modello
# loaded_model = joblib.load('model_random_forest.pkl')

# Funzione per inserire i dati manualmente
colonne = X.columns

def inserisci_dati(colonne):
    dati_input = []
    print("Inserisci i valori per ogni campo (1-10):")
    for colonna in colonne:
        valore = int(input(f"{colonna}: "))
        dati_input.append(valore)
    return [dati_input]

# Inserimento dei dati da parte dell'utente
dati_utente = inserisci_dati(colonne)

# costruiamo il nostro dataframe
df_utente = pd.DataFrame(dati_utente, columns = colonne)

# Previsione basata sui dati inseriti dall'utente
predizione_utente = model.predict(df_utente)
predizione_utente_label = label_encoder.inverse_transform(predizione_utente)

print(f"\n\nLa previsione per i dati inseriti è: {predizione_utente_label[0]}")