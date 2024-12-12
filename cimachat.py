import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Charger les ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Prétraitement
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('french'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

# Charger les objets sauvegardés avec mise en cache
@st.cache_data
def load_model_and_data():
    with open("decision_tree_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    data = pd.read_csv("processed_data.csv")
    return model, vectorizer, data

# Fonction pour obtenir la meilleure réponse
def get_best_response(user_input_cleaned, predicted_livre, data):
    filtered_data = data[data['Livre'] == predicted_livre]

    # Correspondance exacte
    exact_match = filtered_data[filtered_data['Question_cleaned'] == user_input_cleaned]
    if not exact_match.empty:
        return exact_match.iloc[0]['Réponse'], predicted_livre

    # Correspondance similaire dans la catégorie prédite
    input_tokens = set(user_input_cleaned.split())
    for _, row in filtered_data.iterrows():
        question_tokens = set(row['Question_cleaned'].split())
        if input_tokens & question_tokens:
            return row['Réponse'], predicted_livre

    # Correspondance globale si aucune correspondance dans la catégorie
    for _, row in data.iterrows():
        question_tokens = set(row['Question_cleaned'].split())
        if input_tokens & question_tokens:
            return row['Réponse'], row['Livre']

    # Aucune correspondance trouvée
    return "Désolé, je ne trouve pas une réponse appropriée.", "Non spécifié"

# Charger le modèle et les données
model, vectorizer, data = load_model_and_data()

# Application Streamlit
st.title("🤖 Chatbot CIMA - FAQ sur le Code des Assurances")

st.subheader("By Hussein DIALLO")

# Gestion de l'état pour la satisfaction utilisateur
if "satisfaction_given" not in st.session_state:
    st.session_state["satisfaction_given"] = False

# Champ de saisie de la question avec le bouton "Demander"
user_input = st.text_input("💬 Posez votre question :")
if st.button("🔍 Demander"):
    if user_input:
        # Prétraitement de la question utilisateur
        user_input_cleaned = preprocess_text(user_input)
        user_input_vectorized = vectorizer.transform([user_input_cleaned])
        predicted_livre = model.predict(user_input_vectorized)[0]
        response, livre = get_best_response(user_input_cleaned, predicted_livre, data)

        # Affichage de la catégorie prédite en orange
        st.markdown(
            f"<div style='background-color:orange; padding:10px; border-radius:5px;'>"
            f"<strong>Catégorie prédite :</strong> {livre}</div>",
            unsafe_allow_html=True
        )

        # Affichage de la réponse en vert
        st.markdown(
            f"<div style='background-color:lightgreen; padding:10px; border-radius:5px;'>"
            f"<strong>Réponse :</strong> {response}</div>",
            unsafe_allow_html=True
        )

        # Boutons interactifs pour la satisfaction utilisateur
        col1, col2 = st.columns(2)
        with col1:
            if st.button("😊 OUI", key="oui_button"):
                st.session_state["satisfaction_given"] = True
                st.success("Merci pour votre retour !")
        with col2:
            if st.button("😟 NON", key="non_button"):
                st.session_state["satisfaction_given"] = True
                st.error("Nous essayerons d'améliorer nos réponses. Merci pour votre retour.")

        # Vérifier immédiatement l'état pour afficher le bouton "Poser une nouvelle question"
        if st.session_state["satisfaction_given"]:
            st.markdown("---")
            if st.button("🔄 Poser une nouvelle question", key="new_question_button_after_feedback"):
                st.session_state["satisfaction_given"] = False
                st.experimental_rerun()

