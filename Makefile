.PHONY: help setup start stop restart logs clean test build push deploy

# Variables
DOCKER_USERNAME ?= your-dockerhub-username
PROJECT_NAME = mlops
PYTHON = python3
PIP = pip3

help: ## Afficher cette aide
	@echo "Commandes disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup
setup: ## Installation initiale du projet
	@echo "🚀 Installation du projet..."
	cp .env.example .env
	$(PYTHON) -m venv venv
	. venv/bin/activate && $(PIP) install -r requirements.txt
	@echo "✅ Installation terminée! Activez l'environnement: source venv/bin/activate"

setup-dev: ## Installation avec dépendances de développement
	@echo "🔧 Installation mode développement..."
	. venv/bin/activate && $(PIP) install -r requirements-dev.txt
	@echo "✅ Installation dev terminée!"

# Docker Compose
start: ## Démarrer tous les services
	@echo "🚀 Démarrage des services..."
	docker-compose up -d
	@echo "⏳ Attente du démarrage des services..."
	sleep 10
	@echo "✅ Services démarrés!"
	@echo "📊 Airflow: http://localhost:8080 (admin/admin)"
	@echo "📈 MLflow: http://localhost:5000"
	@echo "🗂️  MinIO: http://localhost:9001 (minioadmin/minioadmin)"
	@echo "📉 Prometheus: http://localhost:9090"
	@echo "📊 Grafana: http://localhost:3000 (admin/admin)"

stop: ## Arrêter tous les services
	@echo "🛑 Arrêt des services..."
	docker-compose stop
	@echo "✅ Services arrêtés!"

restart: ## Redémarrer tous les services
	@echo "🔄 Redémarrage des services..."
	docker-compose restart
	@echo "✅ Services redémarrés!"

logs: ## Afficher les logs de tous les services
	docker-compose logs -f

logs-api: ## Afficher les logs de l'API
	docker-compose logs -f api

logs-airflow: ## Afficher les logs d'Airflow
	docker-compose logs -f airflow-webserver airflow-scheduler

clean: ## Arrêter et supprimer tous les conteneurs et volumes
	@echo "🧹 Nettoyage..."
	docker-compose down -v
	rm -rf airflow/logs/*
	@echo "✅ Nettoyage terminé!"

# Database
init-db: ## Initialiser et peupler la base de données
	@echo "💾 Initialisation de la base de données..."
	. venv/bin/activate && $(PYTHON) scripts/populate_database.py
	@echo "✅ Base de données initialisée!"

# Training
train: ## Entraîner le modèle
	@echo "🎓 Entraînement du modèle..."
	. venv/bin/activate && $(PYTHON) src/training/train.py
	@echo "✅ Entraînement terminé!"

# API
run-api: ## Lancer l'API en local
	@echo "🚀 Lancement de l'API..."
	. venv/bin/activate && uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# WebApp
run-webapp: ## Lancer la WebApp en local
	@echo "🌐 Lancement de la WebApp..."
	. venv/bin/activate && streamlit run src/webapp/app.py

# Tests
test: ## Exécuter tous les tests
	@echo "🧪 Exécution des tests..."
	. venv/bin/activate && pytest tests/ -v

test-unit: ## Exécuter les tests unitaires
	@echo "🧪 Tests unitaires..."
	. venv/bin/activate && pytest tests/unit/ -v

test-integration: ## Exécuter les tests d'intégration
	@echo "🧪 Tests d'intégration..."
	. venv/bin/activate && pytest tests/integration/ -v

test-cov: ## Exécuter les tests avec couverture
	@echo "🧪 Tests avec couverture..."
	. venv/bin/activate && pytest tests/ -v --cov=src --cov-report=html
	@echo "📊 Rapport de couverture: htmlcov/index.html"

# Code Quality
lint: ## Vérifier la qualité du code
	@echo "🔍 Vérification du code..."
	. venv/bin/activate && flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
	. venv/bin/activate && black --check src/
	@echo "✅ Code vérifié!"

format: ## Formater le code
	@echo "✨ Formatage du code..."
	. venv/bin/activate && black src/
	. venv/bin/activate && isort src/
	@echo "✅ Code formaté!"

# Docker Build & Push
build: ## Build les images Docker
	@echo "🐳 Build des images Docker..."
	docker build -f Dockerfile.api -t $(DOCKER_USERNAME)/mlops-api:latest .
	docker build -f Dockerfile.webapp -t $(DOCKER_USERNAME)/mlops-webapp:latest .
	@echo "✅ Images buildées!"

push: build ## Push les images sur DockerHub
	@echo "📤 Push des images sur DockerHub..."
	docker push $(DOCKER_USERNAME)/mlops-api:latest
	docker push $(DOCKER_USERNAME)/mlops-webapp:latest
	@echo "✅ Images pushées!"

# Kubernetes
k8s-apply: ## Déployer sur Kubernetes
	@echo "☸️  Déploiement sur Kubernetes..."
	sed -i.bak "s/YOUR_DOCKERHUB_USERNAME/$(DOCKER_USERNAME)/g" k8s/*.yaml
	kubectl apply -f k8s/
	@echo "✅ Déploiement lancé!"
	@echo "📊 Vérification: kubectl get pods -n mlops"

k8s-delete: ## Supprimer le déploiement Kubernetes
	@echo "🗑️  Suppression du déploiement..."
	kubectl delete -f k8s/
	@echo "✅ Déploiement supprimé!"

k8s-status: ## Afficher le statut Kubernetes
	@echo "📊 Statut Kubernetes:"
	kubectl get pods -n mlops
	kubectl get services -n mlops

k8s-logs-api: ## Afficher les logs de l'API sur K8s
	kubectl logs -f -l app=api -n mlops

k8s-logs-webapp: ## Afficher les logs de la WebApp sur K8s
	kubectl logs -f -l app=webapp -n mlops

k8s-port-forward-api: ## Port forward de l'API
	@echo "🔌 Port forward API: http://localhost:8000"
	kubectl port-forward svc/api-service 8000:8000 -n mlops

k8s-port-forward-webapp: ## Port forward de la WebApp
	@echo "🔌 Port forward WebApp: http://localhost:8501"
	kubectl port-forward svc/webapp-service 8501:8501 -n mlops

# Load Testing
load-test: ## Lancer les tests de charge avec Locust
	@echo "🔥 Tests de charge avec Locust..."
	. venv/bin/activate && locust -f scripts/load_test.py --host http://localhost:8000
	@echo "📊 Interface Locust: http://localhost:8089"

# Airflow
airflow-trigger-extraction: ## Trigger le DAG d'extraction de données
	docker-compose exec airflow-webserver airflow dags trigger data_extraction_pipeline

airflow-trigger-training: ## Trigger le DAG d'entraînement
	docker-compose exec airflow-webserver airflow dags trigger continuous_training_pipeline

# Git
git-init: ## Initialiser le repository Git
	@echo "📦 Initialisation Git..."
	git init
	git add .
	git commit -m "Initial commit: MLOps project structure"
	@echo "✅ Git initialisé! Ajoutez le remote: git remote add origin <url>"

# Monitoring
monitoring-ui: ## Ouvrir les interfaces de monitoring
	@echo "📊 Interfaces de monitoring:"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	open http://localhost:9090 || xdg-open http://localhost:9090 || start http://localhost:9090
	open http://localhost:3000 || xdg-open http://localhost:3000 || start http://localhost:3000

# All-in-one
all: setup start init-db ## Setup complet du projet
	@echo "🎉 Projet prêt!"
	@echo ""
	@echo "Prochaines étapes:"
	@echo "1. Activer l'environnement: source venv/bin/activate"
	@echo "2. Lancer l'extraction: make airflow-trigger-extraction"
	@echo "3. Entraîner le modèle: make train"
	@echo "4. Lancer l'API: make run-api"
	@echo "5. Lancer la WebApp: make run-webapp"
