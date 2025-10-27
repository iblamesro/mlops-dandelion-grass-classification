# 🚀 Guide de Démarrage Rapide

## Prérequis

- Docker Desktop avec Kubernetes activé
- Python 3.9+
- Git
- 8 GB RAM minimum

## Installation

### 1️⃣ Cloner le projet

```bash
git clone <votre-repo>
cd MLproject
```

### 2️⃣ Configuration

```bash
# Copier le fichier d'environnement
cp .env.example .env

# Modifier les variables si nécessaire
nano .env
```

### 3️⃣ Lancer les services (Environnement DEV)

```bash
# Démarrer tous les services
docker-compose up -d

# Vérifier que tout fonctionne
docker-compose ps

# Suivre les logs
docker-compose logs -f
```

**Temps de démarrage:** ~2-3 minutes

### 4️⃣ Accéder aux services

| Service | URL | Credentials |
|---------|-----|-------------|
| **Airflow** | http://localhost:8080 | admin / admin |
| **MLflow** | http://localhost:5000 | - |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin |
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | admin / admin |

### 5️⃣ Initialiser la base de données

```bash
# Installer les dépendances Python
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
pip install -r requirements.txt

# Peupler la base de données
python scripts/populate_database.py
```

### 6️⃣ Extraire les données

Dans Airflow (http://localhost:8080):

1. Se connecter (admin/admin)
2. Activer le DAG `data_extraction_pipeline`
3. Cliquer sur "Trigger DAG" ▶️
4. Attendre la fin de l'exécution (peut prendre 10-15 minutes)

### 7️⃣ Entraîner le modèle

**Option A - Via Airflow:**
1. Activer le DAG `model_training_pipeline`
2. Trigger le DAG

**Option B - En local:**
```bash
python src/training/train.py
```

**Durée:** ~20-30 minutes (CPU) / ~5-10 minutes (GPU)

### 8️⃣ Lancer l'API

```bash
# En local
uvicorn src.api.main:app --reload

# Ou via Docker
docker build -f Dockerfile.api -t mlops-api .
docker run -p 8000:8000 mlops-api
```

**Tester l'API:**
```bash
curl http://localhost:8000/health
```

Swagger UI: http://localhost:8000/docs

### 9️⃣ Lancer la WebApp

```bash
# En local
streamlit run src/webapp/app.py

# Ou via Docker
docker build -f Dockerfile.webapp -t mlops-webapp .
docker run -p 8501:8501 mlops-webapp
```

Accès: http://localhost:8501

## 🐳 Déploiement sur Kubernetes

### 1. Préparer les images Docker

```bash
# Build les images
docker build -f Dockerfile.api -t <your-username>/mlops-api:latest .
docker build -f Dockerfile.webapp -t <your-username>/mlops-webapp:latest .

# Push sur DockerHub
docker login
docker push <your-username>/mlops-api:latest
docker push <your-username>/mlops-webapp:latest
```

### 2. Modifier les manifestes K8s

```bash
# Remplacer YOUR_DOCKERHUB_USERNAME par votre username
sed -i '' 's/YOUR_DOCKERHUB_USERNAME/<your-username>/g' k8s/*.yaml
```

### 3. Déployer

```bash
# Appliquer les manifestes
kubectl apply -f k8s/

# Vérifier le déploiement
kubectl get pods -n mlops
kubectl get services -n mlops

# Attendre que tous les pods soient ready
kubectl wait --for=condition=ready pod -l app=api -n mlops --timeout=300s
```

### 4. Accéder aux services

```bash
# API
kubectl port-forward svc/api-service 8000:8000 -n mlops

# WebApp
kubectl port-forward svc/webapp-service 8501:8501 -n mlops
```

## 🧪 Tests

```bash
# Installer les dépendances de test
pip install -r requirements-dev.txt

# Tests unitaires
pytest tests/unit/ -v

# Tests d'intégration
pytest tests/integration/ -v

# Tous les tests avec couverture
pytest --cov=src tests/
```

## 📊 Monitoring

### Prometheus
- URL: http://localhost:9090
- Métriques disponibles: API latency, throughput, model predictions

### Grafana
- URL: http://localhost:3000
- Login: admin / admin
- Dashboards pré-configurés dans `monitoring/grafana/dashboards/`

## 🔄 Continuous Training

Le DAG `continuous_training_pipeline` vérifie automatiquement:
- ✅ Performance du modèle (accuracy < 90%)
- ✅ Nouvelles données (> 100 images)
- ✅ Planning (chaque dimanche 2h)

Pour forcer un retraining:
```bash
# Via Airflow CLI
docker-compose exec airflow-webserver airflow dags trigger continuous_training_pipeline
```

## 🛠️ Troubleshooting

### Services ne démarrent pas
```bash
# Vérifier les logs
docker-compose logs <service-name>

# Redémarrer un service
docker-compose restart <service-name>

# Reconstruire
docker-compose down
docker-compose up --build -d
```

### Problèmes de connexion à la base de données
```bash
# Vérifier que PostgreSQL est prêt
docker-compose exec postgres pg_isready -U mlops_user

# Se connecter à la DB
docker-compose exec postgres psql -U mlops_user -d mlops_db
```

### Erreur MinIO
```bash
# Recréer les buckets
docker-compose restart minio-init
```

### Modèle ne charge pas
```bash
# Vérifier que le modèle existe
ls -la models/

# Télécharger depuis MinIO
mc cp myminio/mlops-models/dandelion_grass_classifier_best.pth models/
```

## 📚 Ressources

- Documentation MLflow: https://mlflow.org/docs/latest/
- Documentation Airflow: https://airflow.apache.org/docs/
- Documentation FastAPI: https://fastapi.tiangolo.com/
- Documentation Kubernetes: https://kubernetes.io/docs/

## 🤝 Support

Pour toute question:
- Créer une issue sur GitHub
- Contacter l'équipe: prillard.martin@gmail.com

## 📝 Checklist avant soumission

- [ ] Tous les tests passent
- [ ] Documentation à jour
- [ ] Images Docker sur DockerHub
- [ ] CI/CD fonctionnel
- [ ] API déployée et accessible
- [ ] WebApp déployée et accessible
- [ ] Monitoring configuré
- [ ] Screenshots dans README.md
- [ ] Liste des membres du groupe dans email
