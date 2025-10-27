# 🌼 MLOps Project: Dandelion vs Grass Classification

## 📝 Description
Ce projet implémente un pipeline MLOps complet pour la classification d'images binaire (pissenlit vs herbe) avec :
- Extraction et prétraitement automatisés des données
- Entraînement et tracking des modèles avec MLflow
- API de prédiction FastAPI
- Interface utilisateur Streamlit
- Déploiement Kubernetes
- Monitoring et Continuous Training

## 👥 Équipe
- [Nom Membre 1]
- [Nom Membre 2]
- [Nom Membre 3]
- [Nom Membre 4]
- [Nom Membre 5]

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   GitHub    │─────▶│ GitHub Actions│─────▶│  DockerHub  │
│             │      │   (CI/CD)     │      │             │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
│  │   API    │  │ Streamlit│  │  MLflow  │  │ Airflow  ││
│  │ FastAPI  │  │  WebApp  │  │          │  │          ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘│
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┤
│  │Prometheus│  │  Grafana │  │     MinIO (S3)           │
│  └──────────┘  └──────────┘  └──────────────────────────┤
└─────────────────────────────────────────────────────────┘
```

## 📚 Stack Technologique

| Composant | Technologie |
|-----------|-------------|
| **Orchestration** | Apache Airflow |
| **ML Framework** | PyTorch / FastAI |
| **Model Registry** | MinIO (S3) |
| **ML Tracking** | MLflow |
| **Database** | PostgreSQL |
| **API** | FastAPI |
| **WebApp** | Streamlit |
| **Containerization** | Docker |
| **Orchestration** | Kubernetes |
| **CI/CD** | GitHub Actions |
| **Monitoring** | Prometheus + Grafana |
| **Load Testing** | Locust |

## 🚀 Installation & Setup

### Prérequis
- Docker Desktop avec Kubernetes activé
- Python 3.9+
- kubectl
- helm (optionnel)

### Environnement DEV (Local)

```bash
# Cloner le repository
git clone <votre-repo>
cd MLproject

# Lancer tous les services avec Docker Compose
docker-compose up -d

# Vérifier que tous les services sont démarrés
docker-compose ps
```

**Services disponibles:**
- Airflow: http://localhost:8080 (admin/admin)
- MLflow: http://localhost:5000
- MinIO: http://localhost:9001 (minioadmin/minioadmin)
- PostgreSQL: localhost:5432
- API: http://localhost:8000
- Streamlit: http://localhost:8501

### Environnement PRODUCTION (Kubernetes)

```bash
# Déployer sur Kubernetes
kubectl apply -f k8s/

# Vérifier les déploiements
kubectl get pods -n mlops

# Accéder aux services
kubectl port-forward svc/api-service 8000:8000 -n mlops
```

## 📊 Utilisation

### 1. Initialiser la base de données

```bash
# Créer la table et insérer les URLs
python scripts/init_database.py
```

### 2. Extraire les données

```bash
# Via Airflow UI: trigger le DAG "data_extraction_pipeline"
# Ou en CLI:
docker-compose exec airflow-webserver airflow dags trigger data_extraction_pipeline
```

### 3. Entraîner le modèle

```bash
# Via Airflow UI: trigger le DAG "model_training_pipeline"
# Ou directement:
python src/training/train.py
```

### 4. Tester l'API

```bash
# Avec curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"

# Ou utiliser Swagger UI: http://localhost:8000/docs
```

### 5. Utiliser la WebApp

Ouvrir http://localhost:8501 et uploader une image

## 🧪 Tests

```bash
# Installer les dépendances de test
pip install -r requirements-dev.txt

# Tests unitaires
pytest tests/unit/

# Tests d'intégration
pytest tests/integration/

# Tests end-to-end
pytest tests/e2e/

# Coverage
pytest --cov=src tests/
```

## 📈 Monitoring

### Prometheus
- Métriques: http://localhost:9090

### Grafana
- Dashboards: http://localhost:3000 (admin/admin)

Dashboards disponibles:
- API Performance
- Model Metrics
- Airflow Pipeline Status
- Infrastructure Health

## 🔄 Continuous Training

Le DAG `continuous_training` s'exécute automatiquement selon les triggers:
- ✅ Nouvelles données disponibles (>100 images)
- ✅ Chaque dimanche à 2h du matin
- ✅ Performance du modèle < 90% d'accuracy

## 🐳 Images Docker

Les images sont disponibles sur DockerHub:
- `<votre-username>/mlops-api:latest`
- `<votre-username>/mlops-webapp:latest`
- `<votre-username>/mlops-training:latest`

## 📁 Structure du Projet

```
MLproject/
├── .github/
│   └── workflows/          # CI/CD GitHub Actions
├── airflow/
│   └── dags/              # DAGs Airflow
├── src/
│   ├── api/               # API FastAPI
│   ├── webapp/            # WebApp Streamlit
│   ├── training/          # Scripts d'entraînement
│   ├── data/              # Preprocessing & feature store
│   └── utils/             # Utilitaires
├── tests/                 # Tests
├── k8s/                   # Manifestes Kubernetes
├── monitoring/            # Prometheus & Grafana config
├── scripts/               # Scripts utilitaires
├── models/                # Modèles sauvegardés
├── docker-compose.yml     # Environnement DEV
├── Dockerfile.api         # Image Docker API
├── Dockerfile.webapp      # Image Docker WebApp
└── requirements.txt       # Dépendances Python
```

## 🎯 Résultats

### Métriques du Modèle
- **Accuracy**: XX%
- **Precision**: XX%
- **Recall**: XX%
- **F1-Score**: XX%

### Performances API
- **Latence moyenne**: XXms
- **Throughput**: XX req/s

### Screenshots

[Ajouter vos screenshots ici]

## 🔧 Choix Techniques

### Pourquoi PyTorch ?
- Flexibilité pour l'architecture du modèle
- Excellente communauté et documentation
- Bon support pour le deployment

### Pourquoi FastAPI ?
- Performance excellente (async)
- Documentation automatique (Swagger)
- Validation des données avec Pydantic

### Pourquoi Airflow ?
- Standard de l'industrie pour orchestration
- UI intuitive
- Extensibilité avec operators personnalisés

## 🤝 Contribution

1. Créer une branche: `git checkout -b feature/nouvelle-fonctionnalite`
2. Commit: `git commit -m "Ajout nouvelle fonctionnalité"`
3. Push: `git push origin feature/nouvelle-fonctionnalite`
4. Créer une Pull Request

## 📧 Contact

Pour toute question: prillard.martin@gmail.com

## 📄 License

Ce projet est réalisé dans le cadre d'un projet éducatif pour AlbertSchool.
