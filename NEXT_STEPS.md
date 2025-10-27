# 📋 ÉTAPES SUIVANTES - Projet MLOps

## ✅ Ce qui a été créé

Votre projet dispose maintenant de :

1. **Structure complète du projet** ✅
   - Dossiers organisés pour le code, tests, scripts
   - Configuration Docker Compose pour DEV
   - Dockerfiles pour API et WebApp

2. **Infrastructure** ✅
   - PostgreSQL pour la base de données
   - MinIO pour le stockage S3
   - MLflow pour le tracking
   - Airflow pour l'orchestration
   - Prometheus + Grafana pour le monitoring

3. **Code source** ✅
   - Modèle PyTorch (ResNet18)
   - API FastAPI avec Swagger
   - WebApp Streamlit
   - DAGs Airflow (extraction, continuous training)
   - Scripts utilitaires

4. **CI/CD** ✅
   - GitHub Actions workflows
   - Manifestes Kubernetes
   - Tests automatisés

5. **Documentation** ✅
   - README complet
   - Guide de démarrage (QUICKSTART.md)
   - Configuration d'exemple (.env.example)

---

## 🚀 PROCHAINES ÉTAPES (dans l'ordre)

### 📅 SEMAINE 1 (27 Oct - 3 Nov)

#### Jour 1 : Setup & Infrastructure
- [ ] **Créer un repository GitHub** pour votre groupe
  ```bash
  cd /Users/sara/Desktop/AlbertSchool/MLOps/MLproject
  git init
  git add .
  git commit -m "Initial commit: MLOps project structure"
  git remote add origin <votre-repo-url>
  git push -u origin main
  ```

- [ ] **Tester Docker Compose**
  ```bash
  # Copier le fichier d'environnement
  cp .env.example .env
  
  # Lancer les services
  docker-compose up -d
  
  # Vérifier les logs
  docker-compose logs -f
  ```

- [ ] **Vérifier l'accès aux services**
  - Airflow : http://localhost:8080
  - MLflow : http://localhost:5000
  - MinIO : http://localhost:9001
  - Prometheus : http://localhost:9090
  - Grafana : http://localhost:3000

#### Jour 2 : Données
- [ ] **Installer les dépendances Python**
  ```bash
  python -m venv venv
  source venv/bin/activate  # Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

- [ ] **Peupler la base de données**
  ```bash
  python scripts/populate_database.py
  ```

- [ ] **Lancer l'extraction des données via Airflow**
  - Activer le DAG `data_extraction_pipeline`
  - Trigger le DAG
  - Surveiller l'exécution
  - Vérifier les images dans MinIO

#### Jour 3 : Entraînement du modèle
- [ ] **Premier entraînement**
  ```bash
  python src/training/train.py
  ```

- [ ] **Vérifier dans MLflow**
  - Voir les métriques
  - Comparer les runs
  - Télécharger le meilleur modèle

- [ ] **Tester le modèle localement**
  ```python
  from src.training.model import load_model
  model = load_model('models/dandelion_grass_classifier_best.pth')
  # Faire quelques prédictions de test
  ```

#### Jour 4 : API & WebApp
- [ ] **Tester l'API localement**
  ```bash
  uvicorn src.api.main:app --reload
  ```
  - Tester avec Swagger : http://localhost:8000/docs
  - Faire des requêtes de test

- [ ] **Tester la WebApp**
  ```bash
  streamlit run src/webapp/app.py
  ```
  - Uploader des images
  - Vérifier les prédictions

- [ ] **Corriger les bugs éventuels**

#### Jour 5 : Tests
- [ ] **Écrire des tests supplémentaires**
  - Tests pour les utilitaires
  - Tests pour les endpoints API
  - Tests end-to-end

- [ ] **Exécuter tous les tests**
  ```bash
  pip install -r requirements-dev.txt
  pytest tests/ -v --cov=src
  ```

- [ ] **Corriger les tests qui échouent**

#### Jour 6 : Docker & DockerHub
- [ ] **Créer un compte DockerHub** si vous n'en avez pas

- [ ] **Build et push des images**
  ```bash
  # Build
  docker build -f Dockerfile.api -t <votre-username>/mlops-api:latest .
  docker build -f Dockerfile.webapp -t <votre-username>/mlops-webapp:latest .
  
  # Push
  docker login
  docker push <votre-username>/mlops-api:latest
  docker push <votre-username>/mlops-webapp:latest
  ```

- [ ] **Tester les images**
  ```bash
  docker run -p 8000:8000 <votre-username>/mlops-api:latest
  docker run -p 8501:8501 <votre-username>/mlops-webapp:latest
  ```

#### Jour 7 : Kubernetes
- [ ] **Activer Kubernetes dans Docker Desktop**
  - Docker Desktop > Settings > Kubernetes > Enable

- [ ] **Modifier les manifestes K8s**
  ```bash
  sed -i '' 's/YOUR_DOCKERHUB_USERNAME/<votre-username>/g' k8s/*.yaml
  ```

- [ ] **Déployer sur Kubernetes**
  ```bash
  kubectl apply -f k8s/
  kubectl get pods -n mlops
  kubectl get services -n mlops
  ```

- [ ] **Tester les services déployés**
  ```bash
  kubectl port-forward svc/api-service 8000:8000 -n mlops
  kubectl port-forward svc/webapp-service 8501:8501 -n mlops
  ```

---

### 📅 AVANT LA DEADLINE (2 Nov)

#### CI/CD
- [ ] **Configurer GitHub Actions**
  - Ajouter les secrets : DOCKER_USERNAME, DOCKER_PASSWORD
  - Tester le workflow CI/CD
  - Vérifier le build automatique

#### Monitoring
- [ ] **Configurer Grafana**
  - Créer des dashboards
  - Ajouter des métriques importantes
  - Prendre des screenshots

#### Documentation
- [ ] **Mettre à jour le README**
  - Ajouter les noms des membres
  - Ajouter les URLs des images Docker
  - Ajouter les screenshots
  - Documenter les choix techniques

- [ ] **Préparer la présentation**
  - 10 min de démo
  - Slides avec architecture
  - Montrer les différents composants

#### Load Testing (optionnel)
- [ ] **Tester avec Locust**
  ```bash
  locust -f scripts/load_test.py --host http://localhost:8000
  ```

#### Continuous Training
- [ ] **Tester le DAG de CT**
  - Trigger manuellement
  - Vérifier le retraining automatique
  - Documenter les triggers

---

## 📝 CHECKLIST FINALE

Avant de soumettre :

- [ ] ✅ Code sur GitHub
- [ ] ✅ README complet avec screenshots
- [ ] ✅ Images Docker sur DockerHub
- [ ] ✅ Tous les tests passent
- [ ] ✅ API accessible (local ou K8s)
- [ ] ✅ WebApp accessible (local ou K8s)
- [ ] ✅ MLflow fonctionnel avec runs
- [ ] ✅ Airflow avec DAGs fonctionnels
- [ ] ✅ Monitoring (Prometheus + Grafana)
- [ ] ✅ CI/CD GitHub Actions configuré
- [ ] ✅ Documentation complète
- [ ] ✅ Liste des membres dans l'email

---

## 📧 EMAIL DE SOUMISSION

À : prillard.martin@gmail.com

**Sujet:** MLOps Project - [Nom du groupe]

**Corps:**

Bonjour,

Nous vous soumettons notre projet MLOps de classification d'images (Dandelion vs Grass).

**Membres du groupe:**
- [Nom 1]
- [Nom 2]
- [Nom 3]
- [Nom 4]
- [Nom 5]

**Liens:**
- Repository GitHub : [URL]
- Images Docker :
  - API : docker.io/[username]/mlops-api:latest
  - WebApp : docker.io/[username]/mlops-webapp:latest

**Environnement de production:**
- Déployé sur : [Kubernetes local / Cloud provider]
- Modèle MLflow : [URL ou screenshots]
- Monitoring : [Screenshots dans README]

**URLs d'accès (si applicable):**
- API : [URL]
- WebApp : [URL]
- MLflow : [URL]
- Grafana : [URL]

Cordialement,
[Nom du groupe]

---

## 🆘 AIDE & SUPPORT

### Problèmes courants

1. **Docker ne démarre pas**
   - Vérifier Docker Desktop est lancé
   - Vérifier les ressources (RAM, CPU)
   - Redémarrer Docker

2. **Services Airflow ne démarrent pas**
   - Attendre 2-3 minutes (initialisation)
   - Vérifier logs : `docker-compose logs airflow-webserver`
   - Recréer : `docker-compose down && docker-compose up -d`

3. **Modèle ne s'entraîne pas**
   - Vérifier que les données sont dans MinIO
   - Vérifier les logs MLflow
   - Réduire BATCH_SIZE si problème de mémoire

4. **Tests échouent**
   - Vérifier les dépendances installées
   - Vérifier les chemins de fichiers
   - Lire les messages d'erreur

### Ressources

- Documentation Docker : https://docs.docker.com/
- Documentation Kubernetes : https://kubernetes.io/docs/
- Documentation Airflow : https://airflow.apache.org/docs/
- Documentation MLflow : https://mlflow.org/docs/
- Documentation FastAPI : https://fastapi.tiangolo.com/
- Documentation PyTorch : https://pytorch.org/docs/

---

## 💡 CONSEILS

1. **Travaillez en équipe** : Divisez les tâches entre les membres
2. **Committez souvent** : Faites des commits réguliers sur GitHub
3. **Testez régulièrement** : Ne pas attendre la fin pour tester
4. **Documentez au fur et à mesure** : N'attendez pas la fin
5. **Préparez la démo** : Entraînez-vous avant la présentation
6. **Screenshots** : Prenez des screenshots de tout ce qui fonctionne

---

## 🎉 BONNE CHANCE !

Vous avez maintenant tous les outils pour réussir ce projet.
Suivez ce plan étape par étape et tout se passera bien !

En cas de blocage, n'hésitez pas à :
- Consulter les logs
- Lire la documentation
- Chercher sur Stack Overflow
- Demander de l'aide à vos coéquipiers

**Deadline : Dimanche 2 Novembre à minuit**

---

*Créé avec ❤️ pour votre succès !*
