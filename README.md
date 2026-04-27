# Blood Cancer Prediction — Flask GUI

## 📁 Folder Structure
```
gui/
├── app.py                  ← Main Flask application
├── requirements.txt        ← Python dependencies
├── templates/
│   └── index.html          ← Web interface
└── model/
    └── blood_cancer_model.keras  ← PUT YOUR MODEL FILE HERE
```

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your model
Copy `blood_cancer_model.keras` (from Person 2) into the `model/` folder.

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
Visit: [http://127.0.0.1:5000](https://blood-cancer-prediction.onrender.com/)

---

## ⚙️ Class Labels
The model predicts 4 classes (update in `app.py` if your dataset folders differ):
- **Benign** — Normal blood cells
- **Early Pre-B ALL** — Early stage leukemia
- **Pre-B ALL** — Pre-B leukemia
- **Pro-B ALL** — Pro-B leukemia

> If your dataset folders have different names, update `CLASS_LABELS` in `app.py` to match the **alphabetical order** of your dataset subfolders.

---

## 🔗 Integration Notes
- Preprocessing: Images are resized to 224×224 and rescaled by 1/255 — matching training exactly.
- Model input shape: `(1, 224, 224, 3)`
- Model output: Softmax over 4 classes
