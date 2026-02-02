# 🧠 Emotion Detection — Facial Expression Recognition

## 📌 Overview
Ce projet implémente un système complet de reconnaissance d’émotions faciales basé sur le **Deep Learning**. 
L’objectif est de prédire automatiquement l’émotion dominante d’un visage à partir d’une image parmi huit catégories :

> **Anger • Contempt • Disgust • Fear • Happy • Neutral • Sad • Surprise**

Nous avons comparé deux stratégies d’entraînement différentes afin d’identifier la plus performante :
1. 🔹 **CNN entraîné from scratch**
2. 🔹 **Transfer Learning** avec **ResNet50** pré-entraîné (ImageNet)

Le modèle final est déployé via une API **FastAPI** et accessible via une interface **Gradio** incluant la détection automatique et le recadrage du visage.

---

## 🚀 Fonctionnalités
* ✅ **Détection automatique** de visage.
* ✅ **Recadrage interactif** pour optimiser la prédiction.
* ✅ **Préprocessing automatique** des images en entrée.
* ✅ **Prédiction en temps réel**.
* ✅ **API REST** robuste via FastAPI.
* ✅ **Interface graphique** intuitive avec Gradio.
* ✅ **Benchmark** complet entre deux approches d'entraînement.

---

## 🧪 Méthodes comparées (Notebooks)
Le projet contient deux notebooks distincts, chacun correspondant à une stratégie spécifique.

### 📘 Notebook 1 — CNN from scratch (`notebook34cdc48808.ipynb`)
**Objectif :** Créer une baseline simple et rapide sans modèle pré-entraîné.



* **Architecture Custom :** Successions de couches `Conv2D`, `MaxPooling`, et couches `Dense` avec activation `Softmax`.
* **Avantages :** Léger, rapide à entraîner et idéal pour comprendre la mécanique des neurones.
* **Limites :** Généralisation limitée et sensibilité aux variations d'éclairage ou de pose.

### 📗 Notebook 2 — Transfer Learning ResNet50 (`notebook_fer.ipynb`)
**Objectif :** Exploiter la puissance d'un modèle de pointe pour maximiser la précision.



* **Pipeline :** Redimensionnement (224×224), passage en RGB et utilisation de `preprocess_input` spécifique à ResNet.
* **Fine-tuning :** Utilisation de ResNet50 (sans la tête de classification) suivi de couches denses personnalisées.
* **Avantages :** Précision supérieure, grande robustesse et excellente capacité de généralisation.
* **Limites :** Modèle plus lourd et temps d'entraînement plus long.

---

## ⚖️ Comparaison rapide

| Critère | CNN Custom | ResNet50 Transfer |
| :--- | :--- | :--- |
| **Complexité** | Faible | Élevée |
| **Temps d'entraînement** | Rapide | Plus long |
| **Mémoire** | Faible | Plus élevée |
| **Robustesse** | Moyenne | Excellente |
| **Précision** | Baseline | Meilleure |
| **Recommandé pour** | Tests / Prototypage | Production |

---

## 🏆 Choix final
Nous avons retenu **ResNet50 + Transfer Learning** pour le modèle de production car il offre :
1. Une **meilleure généralisation** sur des données inconnues.
2. Une **résilience accrue** face aux visages "in the wild" (conditions réelles).
3. Des **performances globales** nettement supérieures à la baseline.

Le CNN *from scratch* est conservé dans le dépôt à titre de référence pédagogique.


## ⚙️ Installation officielle

### 📦 1) Cloner le dépôt
```bash
git clone https://github.com/TCHIKSON/Emotion-Detection.git
cd Emotion-Detection/font/api
```

### 🐍 2) Prérequis Python
Le projet est compatible avec **Python 3.10 → 3.12**.  
*(Évitez Python 3.13 car TensorFlow n'est pas encore totalement stable dessus).*

Vérifiez votre version :

```bash
python --version
```

### 🛠️ 3) Installer les dépendances
```bash
pip install tensorflow fastapi uvicorn gradio pillow numpy opencv-python requests
```

### 📁 4) Structure des fichiers
Pour que le script se lance correctement, vérifiez que le modèle est bien placé à la racine du repo :

```text
Emotion-Detection/
├── best_resnet50_fer_finetuned.keras   # Modèle final
├── class_names.json                    # Noms des émotions
└── font/api/
    ├── main.py                         # Point d'entrée unique
    ├── app.py                          # Serveur API
    └── frontend.py                     # Interface Gradio
```

### ▶️ 5) Lancer le projet
Exécutez la commande suivante depuis le dossier `font/api/` :

```bash
python main.py
```

* **API (Backend) :** http://127.0.0.1:8000/docs  
* **Gradio (Frontend) :** http://127.0.0.1:7860

---

## 🧠 Résumé Rapide

1. **Clone :** `git clone ...`
2. **Install :** `pip install tensorflow fastapi uvicorn gradio pillow numpy opencv-python requests`
3. **Run :** `python main.py`
