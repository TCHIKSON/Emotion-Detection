 Emotion Detection — Facial Expression Recognition
 Overview

Ce projet implémente un système complet de reconnaissance d’émotions faciales basé sur le Deep Learning.

L’objectif est de prédire automatiquement l’émotion dominante d’un visage à partir d’une image parmi :

anger • contempt • disgust • fear • happy • neutral • sad • surprise


Nous avons comparé deux stratégies d’entraînement différentes afin d’identifier la plus performante :

🔹 CNN entraîné from scratch

🔹 Transfer Learning avec ResNet50 pré-entraîné (ImageNet)

Le modèle final est ensuite déployé via une API FastAPI et utilisable via une interface Gradio avec détection automatique de visage + recadrage.

 Fonctionnalités

✅ Détection automatique de visage
✅ Recadrage interactif
✅ Préprocessing automatique
✅ Prédiction temps réel
✅ API REST (FastAPI)
✅ Interface graphique (Gradio)
✅ Deux approches d'entraînement comparées

 Méthodes comparées (Notebooks)

Le projet contient 2 notebooks distincts, chacun correspondant à une stratégie d’entraînement.

 Notebook 1 — CNN from scratch (notebook34cdc48808.ipynb)
 Objectif

Créer une baseline simple et rapide sans modèle pré-entraîné.

Architecture

CNN custom :

Conv2D

MaxPooling

Dense

Softmax

Avantages

✅ Rapide à entraîner
✅ Léger
✅ Facile à comprendre

Limites

❌ Généralisation limitée
❌ Moins robuste aux variations (lumière, pose, bruit)

Usage

Idéal pour :

tests rapides

prototypage

compréhension du dataset

 Notebook 2 — Transfer Learning ResNet50 (notebook_fer.ipynb)
 Objectif

Utiliser un modèle pré-entraîné ImageNet pour améliorer la précision.

Pipeline

Resize → 224×224

RGB

preprocess_input

ResNet50 (include_top=False)

Fine-tuning partiel

Avantages

✅ Meilleure précision
✅ Meilleure robustesse
✅ Excellente généralisation

Limites

❌ Plus lourd
❌ Plus lent à entraîner

Usage

Idéal pour :

performance maximale

production

datasets réels complexes

 Comparaison rapide
Critère	CNN custom	ResNet50 Transfer
Complexité	Faible	Élevée
Temps d'entraînement	Rapide	Plus long
Mémoire	Faible	Plus élevée
Robustesse	Moyenne	Excellente
Précision	Baseline	Meilleure
Recommandé pour	Tests	Production
 Choix final

Nous avons retenu ResNet50 + transfer learning pour le modèle final car :

meilleure généralisation

plus robuste aux visages “in the wild”

meilleures performances globales

Le CNN from scratch sert de baseline comparative pédagogique.
