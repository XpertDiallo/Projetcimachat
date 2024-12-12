import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

# Téléchargement des ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Prétraitement des données
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('french'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

# Charger les données
data = pd.read_json("FAQ_Code_Complet.txt")

# Prétraitement des données
data['Question_cleaned'] = data['Question'].apply(preprocess_text)

# Extraction des catégories (Livres)
def extract_livre(response):
    if "Livre" in response:
        start = response.find("Livre")
        end = response.find(",", start)
        return response[start:end] if end != -1 else response[start:]
    return "Non spécifié"

data['Livre'] = data['Réponse'].apply(extract_livre)

# TF-IDF et division des ensembles
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(data['Question_cleaned'])
y = data['Livre']

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Entraînement du modèle Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=10, min_samples_split=5)
dt_model.fit(X_train, y_train)

# Sauvegarder le modèle, le TF-IDF et les données traitées
with open("decision_tree_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

data.to_csv("processed_data.csv", index=False)
