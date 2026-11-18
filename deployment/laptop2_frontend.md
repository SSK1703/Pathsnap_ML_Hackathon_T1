# Laptop 2 - PathSnap Frontend Deployment

Laptop 2 hosts the evaluator-facing Streamlit application.

## 1. Clone repository

git clone https://github.com/SSK1703/Pathsnap_ML_Hackathon_T1.git

cd Pathsnap_ML_Hackathon_T1

## 2. Create environment

python -m venv .venv

## 3. Activate environment

Windows:

.venv\Scripts\activate

## 4. Install dependencies

pip install -r requirements.txt

pip install -e .

## 5. Configure backend URL

Create .env.

Set:

PATHSNAP_API_URL=http://LAPTOP1_IP:8000

Replace LAPTOP1_IP with the actual IP address of Laptop 1.

## 6. Run frontend

streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501

## 7. Access interface

Open:

http://localhost:8501
