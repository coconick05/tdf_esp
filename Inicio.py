import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# Color fucsia
FUCSIA = "#FF00FF"

st.markdown(
    f"""
    <style>
    /* Todos los textos en fucsia */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp div[data-testid="stMarkdownContainer"] *,
    .stApp [data-testid="stWidgetLabel"] *,
    .stApp [data-testid="stAlert"] *,
    .stApp textarea, .stApp input,
    .stApp button, .stApp button * {{
        color: {FUCSIA} !important;
    }}

    /* Placeholder de los inputs */
    .stApp textarea::placeholder, .stApp input::placeholder {{
        color: {FUCSIA} !important;
        opacity: 0.6;
    }}

    /* Botones: fondo transparente y borde fucsia para que el texto se lea bien */
    .stApp button {{
        background-color: transparent !important;
        border: 1px solid {FUCSIA} !important;
    }}
    .stApp button:hover {{
        background-color: rgba(255, 0, 255, 0.15) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Demo TF-IDF en Español")

# Documentos de ejemplo (NUEVO TEXTO)
default_docs = """El chef prepara una sopa deliciosa en la cocina del restaurante.
La panadera hornea pan fresco muy temprano en la mañana.
Los estudiantes estudian matemáticas en la biblioteca de la universidad.
El astronauta observa las estrellas desde la estación espacial.
Los turistas visitan el museo de arte en el centro de la ciudad.
La doctora atiende a los pacientes en el hospital durante la noche."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    # Minúsculas
    text = text.lower()
    # Solo letras españolas y espacios
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    # Tokenizar
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Layout en dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Documentos (uno por línea):", default_docs, height=150)
    question = st.text_input("❓ Escribe tu pregunta:", "¿Dónde prepara el chef la sopa?")

with col2:
    st.markdown("### 💡 Preguntas sugeridas:")
    
    # NUEVAS preguntas sugeridas sobre el nuevo texto
    if st.button("¿Dónde prepara el chef la sopa?", use_container_width=True):
        st.session_state.question = "¿Dónde prepara el chef la sopa?"
        st.rerun()
    
    if st.button("¿Cuándo hornea el pan la panadera?", use_container_width=True):
        st.session_state.question = "¿Cuándo hornea el pan la panadera?"
        st.rerun()
        
    if st.button("¿Qué estudian los estudiantes en la biblioteca?", use_container_width=True):
        st.session_state.question = "¿Qué estudian los estudiantes en la biblioteca?"
        st.rerun()
        
    if st.button("¿Qué observa el astronauta desde la estación espacial?", use_container_width=True):
        st.session_state.question = "¿Qué observa el astronauta desde la estación espacial?"
        st.rerun()
        
    if st.button("¿Dónde visitan los turistas el museo de arte?", use_container_width=True):
        st.session_state.question = "¿Dónde visitan los turistas el museo de arte?"
        st.rerun()

# Actualizar pregunta si se seleccionó una sugerida
if 'question' in st.session_state:
    question = st.session_state.question

if st.button("🔍 Analizar", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        # Crear vectorizador TF-IDF
        vectorizer =
