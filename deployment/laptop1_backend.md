# Laptop 1 - PathSnap Backend Deployment

Laptop 1 hosts:

- FastAPI backend
- Trained ML classifier
- HMM and Viterbi decoder
- Road-network data
- Server-side reference evaluation data

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

## 5. Configure environment

Copy .env.example to .env.

Configure dataset paths and evaluator input manifest.

## 6. Find Laptop 1 IP

Windows:

ipconfig

Linux:

hostname -I

Identify the IPv4 address of the active network adapter.

## 7. Start API

uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000

## 8. Verify

Open:

http://localhost:8000/health

API documentation:

http://localhost:8000/docs
