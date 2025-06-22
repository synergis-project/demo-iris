# -*- coding: utf-8 -*-

"""
Nom du fichier : app_v2.py
Description : Application web de démonstration V2 pour le traitement d'iris avancé.
              Interface à onglets, options interactives, et pipeline modulaire.
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from roboflow import Roboflow
import os
import tempfile
import io

# --- CONFIGURATION ET CONSTANTES ---
st.set_page_config(page_title="Démo Traitement d'Iris V2", layout="wide")

try:
    ROBOFLOW_API_KEY = st.secrets["ROBOFLOW_API_KEY"]
except (FileNotFoundError, KeyError):
    st.warning("Clé API Roboflow non trouvée. Utilisation d'une clé de secours pour la démo.")
    ROBOFLOW_API_KEY = "JQUGRc1I81vEziqzrvaw" 

PROJET_IRIS = "iris-segmentation-fqgey"
VERSION_IRIS = 5
PROJET_SCLERE = "conuhacks-eye-model-own"
VERSION_SCLERE = 1
CONFIDENCE_THRESHOLD = 0.2

# ==============================================================================
# /// LOGIQUE BACKEND (PIPELINE DE TRAITEMENT D'IMAGE MODULAIRE) ///
# ==============================================================================

@st.cache_resource
def charger_modeles():
    """Charge les modèles IA et les met en cache pour toute la session."""
    if "VOTRE_CLE_API" in ROBOFLOW_API_KEY: return "ERREUR_CLE_API", None
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        model_iris = rf.workspace().project(PROJET_IRIS).version(VERSION_IRIS).model
        model_sclere = rf.workspace().project(PROJET_SCLERE).version(VERSION_SCLERE).model
        return model_iris, model_sclere
    except Exception as e:
        return f"Erreur de connexion à Roboflow : {e}", None

@st.cache_data
def detect_and_mask(_image_bytes):
    """
    Étape 1 du pipeline : Détection et création des masques de base.
    Prend les bytes de l'image pour être compatible avec le cache de Streamlit.
    """
    log = []
    image_pil = Image.open(io.BytesIO(_image_bytes)).convert("RGB")
    image_source = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = image_source.shape[:2]

    model_iris, model_sclere = charger_modeles()
    if model_sclere is None:
        return {"erreur": model_iris}

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_f:
        image_pil.save(temp_f.name, "JPEG")
        temp_file_path = temp_f.name
    
    try:
        log.append("Analyse par le modèle d'iris...")
        preds_iris = model_iris.predict(temp_file_path, confidence=CONFIDENCE_THRESHOLD * 100).json().get('predictions', [])
        log.append(f"-> Trouvé {len(preds_iris)} objets.")
        
        log.append("Analyse par le modèle de sclère...")
        preds_sclere = model_sclere.predict(temp_file_path, confidence=CONFIDENCE_THRESHOLD * 100).json().get('predictions', [])
        log.append(f"-> Trouvé {len(preds_sclere)} objets.")

        if not preds_iris: return {"erreur": "Le modèle d'iris n'a retourné aucune détection fiable."}

        # Logique de consensus
        masque_iris_brut = np.zeros((h, w), dtype=np.uint8)
        preds_iris.sort(key=lambda p: p['width'] * p['height'], reverse=True)
        cv2.fillPoly(masque_iris_brut, [np.array([[pt['x'], pt['y']] for pt in preds_iris[0]['points']], dtype=np.int32)], 255)
        
        masque_oeil_final = masque_iris_brut
        if preds_sclere:
            masque_sclere = np.zeros((h, w), dtype=np.uint8)
            preds_sclere.sort(key=lambda p: p['width'] * p['height'], reverse=True)
            cv2.fillPoly(masque_sclere, [np.array([[pt['x'], pt['y']] for pt in preds_sclere[0]['points']], dtype=np.int32)], 255)
            masque_oeil_final = cv2.bitwise_and(masque_iris_brut, cv2.bitwise_not(masque_sclere))
            log.append("Consensus appliqué.")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        masque_oeil_final = cv2.morphologyEx(masque_oeil_final, cv2.MORPH_OPEN, kernel)
        
        if np.sum(masque_oeil_final) == 0: return {"erreur": "Le consensus des modèles n'a produit aucun masque."}

        masque_pupille = np.zeros((h, w), dtype=np.uint8)
        if len(preds_iris) >= 2:
            contour_pupille = np.array([[pt['x'], pt['y']] for pt in preds_iris[1]['points']], dtype=np.int32)
            if len(contour_pupille) >= 5:
                cv2.ellipse(masque_pupille, cv2.fitEllipse(contour_pupille), 255, -1)
        
        masque_iris_annulus = cv2.bitwise_and(masque_oeil_final, cv2.bitwise_not(masque_pupille))

        return {
            "image_source": image_source,
            "masque_global": masque_oeil_final,
            "masque_annulus": masque_iris_annulus,
            "masque_pupille": masque_pupille,
            "log": log
        }
    finally:
        os.remove(temp_file_path)

def remove_reflections(image, iris_mask):
    """Étape 2a : Suppression des reflets par inpainting."""
    reflections_mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 220, 255)
    reflections_on_iris = cv2.bitwise_and(reflections_mask, iris_mask)
    return cv2.inpaint(image, reflections_on_iris, 3, cv2.INPAINT_NS)

def enhance_texture(image):
    """Étape 2b : Amélioration de la texture par Unsharp Masking."""
    gaussian = cv2.GaussianBlur(image, (0,0), 5.0)
    return cv2.addWeighted(image, 1.8, gaussian, -0.8, 0)

def apply_artistic_effect(base_image, effect, texture_image=None):
    """Étape 3 : Applique l'effet artistique sélectionné."""
    if effect == "Aucun":
        return base_image
    elif effect == "Texture Galaxie" or effect == "Texture Feu":
        texture = cv2.imread(f"textures/{effect.split(' ')[1].lower()}.jpg")
    elif effect == "Texture Personnalisée" and texture_image is not None:
        texture = cv2.cvtColor(np.array(texture_image), cv2.COLOR_RGB2BGR)
    else:
        return base_image

    h, w = base_image.shape[:2]
    texture = cv2.resize(texture, (w, h))
    
    # Superposer la texture sur l'iris
    return cv2.addWeighted(base_image, 0.5, texture, 0.5, 0)


# ==============================================================================
# /// INTERFACE UTILISATEUR (STREAMLIT) ///
# ==============================================================================

st.title("IRIS V2 : Démonstration de Traitement Avancé")

# --- BARRE LATERALE ---
with st.sidebar:
    st.header("Image Source")
    uploaded_eye_file = st.file_uploader("Téléversez une photo d'œil", type=['jpg', 'jpeg', 'png', 'webp', 'jfif'])
    
    if uploaded_eye_file:
        st.image(uploaded_eye_file, caption="Image Originale")

# --- GESTION DES ONGLETS ---
if uploaded_eye_file:
    # Lancer la détection une seule fois et stocker les résultats dans la session
    if "results" not in st.session_state or st.session_state.get("file_id") != uploaded_eye_file.id:
        st.session_state.file_id = uploaded_eye_file.id
        with st.spinner("Analyse IA en cours... Cette étape peut prendre un moment."):
            st.session_state.results = detect_and_mask(uploaded_eye_file.getvalue())

    # Vérifier si la détection a échoué
    if "erreur" in st.session_state.results:
        st.error(f"La détection initiale a échoué : {st.session_state.results['erreur']}")
        with st.expander("Voir le journal de détection"):
            st.write(st.session_state.results.get('log', []))
    else:
        # Si la détection a réussi, afficher les onglets
        tab1, tab2, tab3, tab4 = st.tabs(["Étape 1: Masquage", "Étape 2: Amélioration", "Étape 3: Effets", "Résultat Final"])

        # Extraire les résultats de base
        res = st.session_state.results
        image_source = res['image_source']
        masque_annulus = res['masque_annulus']
        masque_pupille = res['masque_pupille']
        masque_global = res['masque_global']
        
        # --- Onglet 1: Masquage ---
        with tab1:
            st.header("Détection et Détourage Précis par IA")
            st.info("Le contour vert a été généré par un consensus de deux modèles IA pour épouser la forme réelle de l'iris.")
            
            col1, col2 = st.columns(2)
            with col1:
                # Afficher le contour sur l'image
                contours, _ = cv2.findContours(masque_annulus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                img_contour = image_source.copy()
                cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 2)
                st.image(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB), caption="Contour de l'Iris Détecté")
            with col2:
                # Afficher l'iris isolé sur fond transparent
                b, g, r = cv2.split(image_source)
                iris_isole_bgra = cv2.merge([b, g, r, masque_annulus])
                st.image(iris_isole_bgra, caption="Iris Isolé (fond transparent)")

        # --- Onglet 2: Amélioration ---
        with tab2:
            st.header("Nettoyage et Amélioration de la Texture")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Cochez les options pour voir leur effet :")
                do_reflect_removal = st.checkbox("Supprimer les reflets lumineux")
                do_texture_enhance = st.checkbox("Améliorer la texture (netteté)")
            
            # Traiter l'image en fonction des choix
            iris_isole_bgr = cv2.bitwise_and(image_source, image_source, mask=masque_annulus)
            processed_image = iris_isole_bgr.copy()
            if do_reflect_removal:
                processed_image = remove_reflections(processed_image, masque_annulus)
            if do_texture_enhance:
                processed_image = enhance_texture(processed_image)

            with col2:
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Iris amélioré")
            
            # Sauvegarder l'état pour l'onglet suivant
            st.session_state.enhanced_iris = processed_image

        # --- Onglet 3: Effets Artistiques ---
        with tab3:
            st.header("Application d'Effets Artistiques")
            
            # Utiliser l'image améliorée de l'étape précédente
            enhanced_iris = st.session_state.get('enhanced_iris', cv2.bitwise_and(image_source, image_source, mask=masque_annulus))

            effect_choice = st.selectbox("Choisissez un effet :", ["Aucun", "Texture Galaxie", "Texture Feu", "Texture Personnalisée"])
            
            texture_pil = None
            if effect_choice == "Texture Personnalisée":
                uploaded_texture_file = st.file_uploader("Téléversez votre propre texture", type=['jpg', 'jpeg', 'png'])
                if uploaded_texture_file:
                    texture_pil = Image.open(uploaded_texture_file).convert("RGB")
            
            artistic_iris = apply_artistic_effect(enhanced_iris, effect_choice, texture_pil)
            st.image(cv2.cvtColor(artistic_iris, cv2.COLOR_BGR2RGB), caption=f"Iris avec effet '{effect_choice}'")

            # Sauvegarder pour l'onglet final
            st.session_state.artistic_iris = artistic_iris

        # --- Onglet 4: Résultat Final ---
        with tab4:
            st.header("Composition Finale")
            
            # Utiliser l'image de l'étape précédente
            artistic_iris = st.session_state.get('artistic_iris', cv2.bitwise_and(image_source, image_source, mask=masque_annulus))
            
            # Assemblage
            fond_troue = cv2.bitwise_and(image_source, image_source, mask=cv2.bitwise_not(masque_global))
            pupille_originale = cv2.bitwise_and(image_source, image_source, mask=masque_pupille)
            nouvel_oeil = cv2.add(artistic_iris, pupille_originale)
            image_finale = cv2.add(fond_troue, nouvel_oeil)

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB), caption="Image Originale")
            with col2:
                st.image(cv2.cvtColor(image_finale, cv2.COLOR_BGR2RGB), caption="Résultat Final Intégré")

            # Bouton de téléchargement
            result_as_bytes = cv2.imencode('.png', cv2.cvtColor(image_finale, cv2.COLOR_BGR2RGB))[1].tobytes()
            st.download_button("Télécharger le Résultat Final", result_as_bytes, "iris_final.png", "image/png")

else:
    st.info("👋 Bienvenue ! Veuillez téléverser une photo d'œil dans la barre latérale pour commencer la démonstration.")
