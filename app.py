# -*- coding: utf-8 -*-

"""
Nom du fichier : app.py (Version Finale pour Déploiement)
Description : Version finale de l'application Streamlit.
              - Gestion sécurisée de la clé API pour le déploiement.
              - Interface interactive avec sliders pour les paramètres.
              - Logique de détection robuste avec plans de secours.
              - Gestion du cache corrigée et optimisée avec @st.cache_data.
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from roboflow import Roboflow
import os
import tempfile
import io

# --- CONFIGURATION SÉCURISÉE ---
try:
    ROBOFLOW_API_KEY = st.secrets["ROBOFLOW_API_KEY"]
except (FileNotFoundError, KeyError):
    st.warning("Clé API Roboflow non trouvée dans les secrets. L'application pourrait ne pas fonctionner après déploiement.")
    ROBOFLOW_API_KEY = "VOTRE_CLE_API_POUR_TEST_LOCAL" 

PROJET_IRIS = "iris-segmentation-fqgey"
VERSION_IRIS = 5
PROJET_SCLERE = "conuhacks-eye-model-own"
VERSION_SCLERE = 1

# ==============================================================================
# /// LOGIQUE BACKEND (PIPELINE DE TRAITEMENT D'IMAGE) ///
# ==============================================================================

@st.cache_resource
def charger_modeles():
    """Charge les deux modèles IA depuis Roboflow et les met en cache."""
    if "VOTRE_CLE_API" in ROBOFLOW_API_KEY:
        return "ERREUR_CLE_API", None
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        model_iris = rf.workspace().project(PROJET_IRIS).version(VERSION_IRIS).model
        model_sclere = rf.workspace().project(PROJET_SCLERE).version(VERSION_SCLERE).model
        return model_iris, model_sclere
    except Exception as e:
        return f"Erreur de connexion à Roboflow : {e}", None

def get_predictions_from_model(model, image_path, confidence_threshold):
    """Appelle un modèle et retourne les prédictions filtrées."""
    try:
        result = model.predict(image_path, confidence=confidence_threshold * 100).json()
        return [p for p in result.get('predictions', []) if p.get('confidence', 0) > confidence_threshold]
    except Exception as e:
        print(f"Avertissement: Erreur lors de l'appel au modèle: {e}")
        return []

def create_cleaned_mask_from_preds(preds, image_shape):
    """Crée un masque nettoyé à partir de la plus grande prédiction."""
    h, w = image_shape[:2]
    if not preds: return None
    preds.sort(key=lambda p: p['width'] * p['height'], reverse=True)
    mask = np.zeros((h, w), dtype=np.uint8)
    contour = np.array([[pt['x'], pt['y']] for pt in preds[0]['points']], dtype=np.int32)
    cv2.fillPoly(mask, [contour], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# --- CORRECTION : Ajout de @st.cache_data et renommage des arguments ---
@st.cache_data
def process_iris_image(image_oeil_bytes, image_texture_bytes, confidence_threshold, transparency_level):
    """
    Pipeline de traitement principal. Mis en cache par Streamlit.
    Ne sera ré-exécuté que si un des arguments change.
    """
    log = []
    temp_eye_file_path = None
    
    # On convertit les bytes en objet PIL ici, à l'intérieur de la fonction cachée
    image_oeil_pil = Image.open(io.BytesIO(image_oeil_bytes)).convert("RGB")
    image_texture_pil = Image.open(io.BytesIO(image_texture_bytes)).convert("RGB")

    try:
        log.append("Chargement des modèles d'IA (depuis le cache si possible)...")
        model_iris, model_sclere = charger_modeles()
        if model_sclere is None:
            log.append(model_iris)
            return None, log

        log.append("Création d'un fichier temporaire...")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_f:
            image_oeil_pil.save(temp_f.name, "JPEG")
            temp_eye_file_path = temp_f.name
        
        image_oeil_source = cv2.cvtColor(np.array(image_oeil_pil), cv2.COLOR_RGB2BGR)
        image_texture = cv2.cvtColor(np.array(image_texture_pil), cv2.COLOR_RGB2BGR)
        h, w = image_oeil_source.shape[:2]

        log.append(f"Étape 1 : Détection avec seuil à {confidence_threshold:.3f}...")
        preds_iris = get_predictions_from_model(model_iris, temp_eye_file_path, confidence_threshold)
        preds_sclere = get_predictions_from_model(model_sclere, temp_eye_file_path, confidence_threshold)
        
        # ... la logique de fallback reste identique ...
        masque_oeil_final = None
        if preds_iris and preds_sclere:
            masque_vote_iris = create_cleaned_mask_from_preds(preds_iris, (h,w))
            masque_vote_sclere = create_cleaned_mask_from_preds(preds_sclere, (h,w))
            if masque_vote_iris is not None and masque_vote_sclere is not None:
                masque_oeil_final = cv2.bitwise_and(masque_vote_iris, cv2.bitwise_not(masque_vote_sclere))
        
        if masque_oeil_final is None or np.sum(masque_oeil_final) == 0:
            if preds_iris: masque_oeil_final = create_cleaned_mask_from_preds(preds_iris, (h,w))

        if masque_oeil_final is None or np.sum(masque_oeil_final) == 0:
             if preds_sclere:
                masque_sclere = create_cleaned_mask_from_preds(preds_sclere, (h,w))
                if masque_sclere is not None:
                    contours, _ = cv2.findContours(cv2.bitwise_not(masque_sclere), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        masque_oeil_final = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(masque_oeil_final, [max(contours, key=cv2.contourArea)], -1, 255, -1)

        if masque_oeil_final is None or np.sum(masque_oeil_final) == 0:
            log.append("ERREUR : Aucune stratégie n'a pu produire un masque valide.")
            return None, log

        # ... la suite de la logique est inchangée ...
        masque_pupille = np.zeros((h, w), dtype=np.uint8)
        masque_iris_annulus = masque_oeil_final
        if len(preds_iris) >= 2:
            preds_iris.sort(key=lambda p: p['width'] * p['height'], reverse=True)
            contour_pupille = np.array([[pt['x'], pt['y']] for pt in preds_iris[1]['points']], dtype=np.int32)
            if len(contour_pupille) >= 5:
                cv2.ellipse(masque_pupille, cv2.fitEllipse(contour_pupille), 255, -1)
                masque_iris_annulus = cv2.bitwise_and(masque_oeil_final, cv2.bitwise_not(masque_pupille))

        log.append("Étape 3 : Composition et effet artistique...")
        texture_redimensionnee = cv2.resize(image_texture, (w, h))
        iris_original = cv2.bitwise_and(image_oeil_source, image_oeil_source, mask=masque_iris_annulus)
        texture_isolee = cv2.bitwise_and(texture_redimensionnee, texture_redimensionnee, mask=masque_iris_annulus)
        iris_texture_transparente = cv2.addWeighted(iris_original, 1 - transparency_level, texture_isolee, transparency_level, 0)
        fond_troue = cv2.bitwise_and(image_oeil_source, image_oeil_source, mask=cv2.bitwise_not(masque_oeil_final))
        pupille_originale = cv2.bitwise_and(image_oeil_source, image_oeil_source, mask=masque_pupille)
        nouvel_oeil = cv2.add(iris_texture_transparente, pupille_originale)
        image_finale_bgr = cv2.add(fond_troue, nouvel_oeil)
        
        image_finale_rgb = cv2.cvtColor(image_finale_bgr, cv2.COLOR_BGR2RGB)
        log.append("Traitement terminé avec succès !")
        return image_finale_rgb, log

    finally:
        if temp_eye_file_path and os.path.exists(temp_eye_file_path):
            os.remove(temp_eye_file_path)

# ==============================================================================
# /// INTERFACE UTILISATEUR (STREAMLIT) ///
# ==============================================================================

st.set_page_config(page_title="Démo Traitement d'Iris", layout="wide")
st.title("Démonstration de Traitement Artistique d'Iris")

with st.sidebar:
    st.header("1. Paramètres")
    confidence_slider = st.slider("Seuil de Confiance IA", 0.01, 0.99, 0.1, 0.01)
    transparency_slider = st.slider("Opacité de la Texture", 0.0, 1.0, 0.45, 0.05)
    st.header("2. Fichiers")
    uploaded_eye_file = st.file_uploader("Image de l'œil", type=['jpg', 'jpeg', 'png', 'webp', 'jfif'])
    uploaded_texture_file = st.file_uploader("Image de Texture", type=['jpg', 'jpeg', 'png', 'webp', 'jfif'])

if uploaded_eye_file and uploaded_texture_file:
    if st.button("Lancer le Traitement", type="primary", use_container_width=True):
        eye_bytes = uploaded_eye_file.getvalue()
        texture_bytes = uploaded_texture_file.getvalue()
        
        with st.spinner("Analyse en cours par l'IA..."):
            # On passe les bytes à la fonction cachée
            final_image, log = process_iris_image(eye_bytes, texture_bytes, confidence_slider, transparency_slider)
        
        if final_image is None:
            st.error("Le traitement a échoué.")
        else:
            st.success("Traitement réussi !")
            st.subheader("Comparaison Avant / Après")
            col1, col2 = st.columns(2)
            with col1:
                st.image(eye_bytes, caption="Image Originale", use_column_width=True)
            with col2:
                st.image(final_image, caption="Résultat Final", use_column_width=True)

        with st.expander("Afficher le journal de traitement"):
            for entry in log:
                if "ERREUR" in entry or "Échec" in entry: st.error(entry)
                else: st.info(entry)
else:
    st.info("Veuillez téléverser une image de l'œil et une image de texture pour commencer.")