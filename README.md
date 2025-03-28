# Reconstruction d’images en tomographie photoacoustique avec régularisation combinée variation totale - Cauchy

## Contexte
La tomographie photoacoustique (PAT) est une technique d’imagerie biomédicale pour laquelle la reconstruction des images est exigeante numériquement. Nous proposons une méthode de reconstruction reposant sur la minimisation d’une fonction coût utilisant une régularisation de type Cauchy appliquée sur la norme du gradient, mimant ainsi la variation totale. Elle est minimisée à l’aide d’un algorithme de BFGS modifié tenant compte de sa non-convexité et offrant une convergence rapide. Sur une expérience numérique simple, dans un contexte PAT, nous montrons que cette nouvelle régularisation mène à une reconstruction de meilleure qualité que celle obtenue par variation totale et ce, en un temps de calcul d’un ordre de grandeur plus rapide.

## Installation
Assurez-vous d'avoir Python installé, puis utilisez la commande suivante pour installer les dépendances nécessaires :
`pip install -r requirements.txt`

## Utilisation
1. Génération du modèle :

Exécutez le script suivant pour créer les modèles nécessaires à la reconstruction :
`python model_reconstruction.py`

2. Lancer les tests de reconstruction :

Ouvrez et exécutez le fichier `reconstruction_test.py` cellule par cellule. Le script est structuré avec des marqueurs `#%%`, permettant son exécution progressive comme un notebook.

Les résultats seront sauvegardés dans le dossier `resultats/`.



