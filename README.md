# DeepLeaf AI
Plant Disease Detection Using Convolutional Neural Networks (CNN)

## Features
- Dataset preparation and reproducible train/validation/test split
- Custom CNN training with augmentation, early stopping, and checkpointing
- Evaluation: accuracy, precision, recall, F1, confusion matrix
- Flask REST API for image prediction
- SQLite prediction history
- Responsive HTML/CSS/JS frontend
- Grad-CAM explainability endpoint
- Optional treatment/advisory information

## 1. Setup
```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Dataset
Download PlantVillage (or an equivalent labeled leaf dataset). Put class folders under `dataset/raw/`, for example:
```
dataset/raw/Tomato___healthy/*.jpg
dataset/raw/Tomato___Early_blight/*.jpg
```
The included `src/prepare_dataset.py` creates stratified-by-class train/validation/test folders. No dataset is included in this repository.

## 3. Prepare data
```bash
python src/prepare_dataset.py --input dataset/raw --output dataset/processed --train 0.70 --val 0.15 --test 0.15
```

## 4. Train CNN
```bash
python src/train.py --data dataset/processed --epochs 20 --batch-size 32
```
The model and class names are saved in `models/`.

## 5. Evaluate
```bash
python src/evaluate.py --data dataset/processed --model models/deepleaf_cnn.keras
```
Reports and confusion matrix are written to `reports/`.

## 6. Run web app
```bash
python backend/app.py
```
Open http://127.0.0.1:5000

## Project structure
See `docs/ARCHITECTURE.md` and `docs/PROJECT_REPORT.md`.

## Notes
The model is an educational/portfolio classifier and should not be treated as a definitive agricultural diagnosis. Real-world field images can differ substantially from curated datasets, so test with external images before making claims about deployment accuracy.
