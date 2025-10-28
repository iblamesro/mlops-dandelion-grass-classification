# 📦 Téléchargement des Modèles et Données

## 🎯 Fichiers Exclus de Git

Pour garder le repository léger (<100 MB), les fichiers suivants sont exclus de Git :

### 1. 🤖 Modèles (256 MB total)
- `models/best_model.pth` (128 MB)
- `models/downloaded/model_from_s3.pth` (128 MB)

### 2. 📊 MLflow Runs (43 MB)
- `mlruns/` (expériences et artifacts)

## 🚀 Comment Récupérer les Fichiers

### Option 1 : Téléchargement Direct (Recommandé)

**Liens de téléchargement :**
- Modèle principal : [À ajouter - Google Drive / Dropbox]
- Backup S3 : Voir script `scripts/download_model_from_s3.py`

```bash
# Télécharger depuis S3/MinIO
python3 scripts/download_model_from_s3.py
```

### Option 2 : Ré-entraîner le Modèle

```bash
# Entraîner un nouveau modèle
python3 scripts/train_advanced.py
```

### Option 3 : Utiliser l'API en Production

L'API charge automatiquement le modèle depuis S3 si absent :
```bash
python3 run_api.py
```

## 📁 Structure Après Téléchargement

```
MLproject/
├── models/
│   ├── best_model.pth          ← 128 MB (télécharger)
│   ├── model_metadata.json     ✅ (dans git)
│   └── downloaded/
│       └── model_from_s3.pth   ← 128 MB (optionnel)
├── mlruns/                     ← Sera recréé lors du training
└── data/
    ├── dandelion/              ✅ (images dans git)
    └── grass/                  ✅ (images dans git)
```

## 🔄 Workflow Recommandé

### Pour Développement Local :
1. Cloner le repo
2. Télécharger `best_model.pth`
3. Placer dans `models/`
4. Lancer l'API ou WebApp

### Pour Production :
1. Cloner le repo
2. Configurer S3/MinIO
3. L'app télécharge automatiquement

### Pour Training :
1. Cloner le repo
2. Les images sont déjà présentes
3. Lancer `scripts/train_advanced.py`
4. Nouveau modèle créé dans `models/`

## 📝 Notes Importantes

- **GitHub limite** : 100 MB par fichier
- **Solution choisie** : Exclusion des gros fichiers + téléchargement externe
- **Alternative** : Git LFS (coût supplémentaire)
- **Images** : Gardées dans git (15 MB total)

## 🆘 En Cas de Problème

Si le modèle est manquant :
```bash
# Vérifier
ls -lh models/best_model.pth

# Si absent, télécharger
python3 scripts/download_model_from_s3.py

# Ou ré-entraîner
python3 scripts/train_advanced.py
```

## 📧 Contact

Questions ? prillard.martin@gmail.com
