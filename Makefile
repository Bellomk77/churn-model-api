.PHONY: setup train serve test simulate docker
setup: ; pip install -r requirements.txt && python scripts/download_data.py
train: ; python -m model.train
serve: ; uvicorn app.main:app --reload
test: ; pytest -q
simulate: ; python scripts/simulate_traffic.py
docker: ; docker build -t churn-model-api . && docker run -p 8000:8000 churn-model-api
