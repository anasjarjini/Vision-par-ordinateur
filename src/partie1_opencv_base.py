"""
Partie 1 : Concepts de base de la vision par ordinateur
Exercice 1 : Analyse d’une image
- Chargement d'une image avec OpenCV
- Conversion en niveaux de gris
- Flou Gaussien (réduction de bruit)
- Détection de contours avec l'algorithme de Canny
"""

import cv2
import os
import matplotlib.pyplot as plt

def main():
    image_path = os.path.join("images", "image1.jpg")
    if not os.path.exists(image_path):
        image_path = os.path.join("images", "tiger.jpg")

    print(f"Chargement de l'image : {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")

    # Propriétés de l'image
    h, w, c = img_bgr.shape
    print(f"Dimensions : Hauteur={h}px, Largeur={w}px, Canaux={c}")
    print(f"Type des données (dtype) : {img_bgr.dtype}")

    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Flou Gaussien optionnel pour réduire le bruit haute fréquence
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détection de contours (Canny)
    # 100 = seuil bas (hysteresis low), 200 = seuil haut (hysteresis high)
    edges = cv2.Canny(blur, 100, 200)

    # Sauvegarde des résultats
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/exo1_gris.jpg", gray)
    cv2.imwrite("outputs/exo1_contours_canny.jpg", edges)
    print("Images sauvegardées dans le dossier 'outputs/'.")

    # Affichage avec Matplotlib (compatible environnement sans affichage X11)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Image Originale (RGB)")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("2. Niveaux de Gris")
    axes[1].axis("off")

    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title("3. Contours de Canny (100, 200)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/exo1_resultat_global.png")
    print("Figure récapitulative enregistrée sous 'outputs/exo1_resultat_global.png'.")

if __name__ == "__main__":
    main()
