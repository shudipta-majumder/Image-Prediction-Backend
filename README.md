# 🧠 Django ML Project

This is a **Django-based Machine Learning project** that provides REST APIs for training, serving, and monitoring ML models.  
The project integrates **Django + Django REST Framework + ML (scikit-learn / TensorFlow / PyTorch)** for end-to-end ML workflows.

---

## 🚀 Features
- Django backend with REST APIs
- ML model training & prediction endpoints
- Model versioning and storage
- Async task support with Celery & Redis (optional)
- Swagger / DRF-YASG API documentation
- Dockerized for easy deployment

---

## 📦 Tech Stack
- **Backend:** Django, Django REST Framework  
- **ML Framework:** scikit-learn / TensorFlow / PyTorch  
- **Database:** PostgreSQL / SQLite (for dev)  
- **Task Queue (Optional):** Celery + Redis  
- **Deployment:** Docker, Gunicorn, Nginx  

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/shudipta-majumder/Image-Prediction-Backend.git
cd django-ml-project
2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Apply Migrations
python manage.py migrate
5️⃣ Run Development Server
python manage.py runserver
Project will be available at: 👉 http://127.0.0.1:8000

🤖 ML Workflow
Train Model
python manage.py train_model
Predict via API
POST request to /api/predict/
Example:
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
Example Response
{
  "prediction": "Iris-setosa",
  "confidence": 0.94,
  "model_version": "v1.0.0"
}
🧪 Running Tests

python manage.py test
📜 Environment Variables
Create a .env file in project root:
DEBUG=True
SECRET_KEY=your_secret_key
DATABASE_URL=postgres://user:password@localhost:5432/db_name
REDIS_URL=redis://localhost:6379/0
🐳 Docker Setup
docker-compose up --build
📖 API Documentation
After running the server, visit:
👉 http://127.0.0.1:8000/swagger/

📌 Project Structure
django-ml-project/
│── ml/                 # ML training & prediction code
│── api/                # Django REST API endpoints
│── core/               # Django project configs
│── manage.py
│── requirements.txt
│── README.md
