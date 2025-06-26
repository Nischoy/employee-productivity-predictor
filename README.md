# Employee Performance Predictor

Flask web app + XGBoost model that predicts Employee productivity.

## Folders

* `flask_app/` – Flask web application  
  * `app.py` – main Flask server  
  * `templates/` – HTML pages (`home.html`, `about.html`, `predict.html`, `submit.html`)  
  * `XGBoost.pkl` – trained prediction model  

* `model_training/` – data-cleaning and model-building code  
  * `model_training.py` – full pipeline script (converted from the original notebook)  

* `dataset/`  
  * `garments_worker_productivity.csv` – raw dataset used to train the model  

## Quick start
```bash
pip install -r requirements.txt
python app.py 
