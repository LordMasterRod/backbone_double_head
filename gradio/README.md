---
title: Dermato-IA — démonstrateur de recherche
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# Déployer la démonstration sur Hugging Face Spaces

Cette application charge le classifieur médical à une seule tête du notebook
(EfficientNet-B0, sept classes). Le checkpoint fourni n’entraîne pas de tête
de détection des attaques : FGSM est proposé uniquement comme simulation de
sensibilité du classifieur, sans prétention de détection.

## Préparer le checkpoint

Le fichier reçu est déjà une archive de sérialisation PyTorch. Il faut le copier
tel quel, sans l’extraire, sous le nom attendu par l’application :

```powershell
New-Item -ItemType Directory -Force models
Copy-Item "C:\Users\xphones\Downloads\best_backbone_model.zip" `
  ".\models\best_backbone_model.pth"
```

Le chargement est strict (`weights_only=True`, `strict=True`) : un checkpoint
incompatible ou incomplet empêchera le démarrage au lieu de produire des
résultats trompeurs.

## Publier la Space

1. Créez une Space Gradio sur Hugging Face et choisissez sa visibilité.
2. Installez Git LFS si nécessaire, puis clonez l’adresse Git indiquée par la
   Space.
3. Copiez `app.py`, `requirements.txt`, ce `README.md` et le dossier `models`
   dans le dépôt cloné.
4. Depuis ce dépôt, exécutez :

```powershell
git lfs install
git add app.py requirements.txt README.md models/best_backbone_model.pth
git commit -m "Add dermatoscopic classification demo"
git push
```

Le fichier de poids est suivi par Git LFS grâce au fichier `.gitattributes`
fourni. N’inscrivez jamais de jeton Hugging Face dans le code, les fichiers ou
l’historique Git ; utilisez l’authentification locale de Hugging Face CLI si la
Space n’est pas publique.

La première construction installe les dépendances et charge le modèle sur CPU.
Vérifiez les journaux de construction de la Space : toute erreur de
compatibilité du checkpoint ou de dépendance y apparaîtra.
