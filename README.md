# Obesity Level Prediction API

An end-to-end machine learning project that classifies a person's obesity level from lifestyle and body measurements, served through a FastAPI backend and a Streamlit web application.

[Video Walkthrough](https://youtu.be/YpMffL-X2fI)

## Problem

A health clinic focused on obesity prevention wants to reach high-risk people earlier, with education or intervention before the problem becomes severe. Campaigns work better when they are aimed at the right people, so the clinic needs a way to estimate someone's obesity level from everyday habits.

This project trains a classifier that predicts one of seven obesity levels, and deploys it as two separate services: a backend API that owns the model and preprocessing, and a front end that only collects input and shows the result. Keeping them separate means other systems in the clinic could call the same API later without going through the web page.

## Dataset

`ObesityDataSet2.csv` contains 1,056 records with 16 input features and one target. It is a course-provided variant of the public obesity levels dataset, which is based on eating habits and physical condition.

| Category | Features |
|---|---|
| Body | Gender, Age, Height, Weight |
| Eating habits | FAVC (high-calorie food), FCVC (vegetable frequency), NCP (main meals), CAEC (snacking), CH2O (water intake), SCC (calorie monitoring) |
| Lifestyle | SMOKE, FAF (physical activity), TUE (technology use), CALC (alcohol), MTRANS (transport) |
| History | family_history_with_overweight |

**Target:** `NObeyesdad`, with seven classes: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III.

The data was not perfectly clean:

| Issue | Detail |
|---|---|
| Missing values | FCVC: 37, MTRANS: 26 |
| Outliers (IQR rule) | NCP: 257 rows flagged, Weight: 1 row |
| Age stored as text | The column was read as text rather than numbers |

## Approach

### Modelling

1. **Feature engineering.** Added `BMI = Weight / Height²`.
2. **Split.** 80/20 train/test with a fixed random seed. All preprocessing statistics are computed from the training set only, then applied to the test set, to avoid leakage.
3. **Missing values.** Numeric columns were filled with the training median and categorical columns with the training mode.
4. **Outliers.** Clipped to the IQR bounds (winsorizing) so no rows were dropped.
5. **Encoding.** Categorical columns use `LabelEncoder`; the target has its own encoder. Every fitted encoder is saved.
6. **Scaling.** `StandardScaler` on the numeric columns, also saved.
7. **Model comparison.** Random Forest and XGBoost were trained on the same split, and the one with the higher accuracy was saved as the final model.

### Deployment

```
┌─────────────┐   POST /predict (JSON)   ┌──────────────────────────┐
│  Streamlit  │ ───────────────────────► │  FastAPI                 │
│  (form UI)  │                          │  BMI → encode → scale →  │
│             │ ◄─────────────────────── │  Random Forest → label   │
└─────────────┘   {"prediction": ...}    └──────────────────────────┘
```

The API loads the model, scaler and encoders once at startup. For each request it validates the input with a Pydantic schema, rebuilds the BMI feature, applies the same encoding and scaling used in training, and returns the predicted label. The Streamlit app provides a form for all inputs and shows the result.

## Results

| Model | Test accuracy |
|---|---|
| Random Forest (selected) | 98.6% |
| XGBoost | 97.2% |

Both models scored between 0.93 and 1.00 precision and recall across all seven classes on a test set of 212 records. Random Forest was selected as the higher scorer and saved as `best_model_random_forest.pkl`.

**How to read the 98.6%.** The obesity levels in this dataset are defined largely by BMI, and BMI is computed from the Height and Weight inputs. The model is therefore close to learning a rule that is already contained in its own features. The accuracy shows that the pipeline works correctly, but it should not be read as evidence that the model could predict obesity from lifestyle habits alone.

### Test cases

Two end-to-end test cases were run through the Streamlit app with the API running.

| Case | Input (key fields) | BMI | Prediction |
|---|---|---|---|
| 1 | Male, 20 years, 1.73 m, 67 kg | 22.4 | Normal_Weight |
| 2 | Female, 55 years, 1.56 m, 72 kg | 29.6 | Overweight_Level_I |

![Test case 1](screenshots/TestCase1.jpg)
![Test case 2](screenshots/TestCase2.jpg)

## Repository Structure

```
├── ObesityDataSet2.csv        # Raw dataset
├── modelling.ipynb            # EDA, preprocessing, model comparison, saving artifacts
├── api.py                     # FastAPI backend (POST /predict)
├── app.py                     # Streamlit front end
├── model/
│   ├── best_model_random_forest.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── target_encoder.pkl
├── screenshots/               # Test case results
├── requirements.txt
├── LICENSE
└── README.md
```

## Running Locally

```bash
git clone https://github.com/SeafoodEnthusiast/obesity-prediction-fastapi-streamlit.git
cd obesity-prediction-fastapi-streamlit
pip install -r requirements.txt
```

Start the backend first, from the repository root so the `model/` folder is found:

```bash
uvicorn api:app --reload
```

Then, in a second terminal, start the front end:

```bash
streamlit run app.py
```

The Streamlit app expects the API at `http://localhost:8000`. The interactive API docs are at `http://localhost:8000/docs`.

You can also call the API directly:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male", "Age": 20, "Height": 1.73, "Weight": 67,
    "family_history_with_overweight": "yes", "FAVC": "no", "FCVC": 2,
    "NCP": 3, "CAEC": "Sometimes", "SMOKE": "no", "CH2O": 3, "SCC": "no",
    "FAF": 2, "TUE": 3, "CALC": "no", "MTRANS": "Walking"
  }'
```

## Testing and Monitoring Notes

For the production side of the project, I wrote up a testing plan and a drift discussion. The short version:

**What to test before going live**

- **Input validation.** Reject impossible values such as a zero or negative Age, or a Height and Weight outside a plausible range.
- **Unexpected categories.** A SMOKE value like "maybe" or "YES" in capitals, or a MTRANS value outside the list, should be handled by a fallback rather than crash the prediction.
- **End to end.** Check the whole path from the Streamlit form to the API, through the model, and back to the screen.
- **Extreme cases.** Very high BMI or very unhealthy habits should still return a stable prediction.

**What to watch after launch**

- **Data drift** is when the inputs the system receives start to differ from the training data, for example many more users under 18 or over 40 than the model was trained on.
- **Model drift** is when the relationship between habits and obesity level itself changes, so a model trained earlier slowly becomes less accurate.
- The fix for both is to monitor accuracy and input distributions regularly, and to retrain on newer data when they slip.

## Notes on Some Decisions

I saved the scaler and all the encoders next to the model, not just the model itself. The API has to transform a user's input exactly the way the training data was transformed, otherwise a category could map to a different number and the model would return a wrong answer without raising any error.

I computed BMI inside the API as well as in the Streamlit app. The API is the part that has to be correct no matter who calls it, so it should not trust a client to send a derived feature.

Looking back at this project, these are the things I would change first:

- **Fix Age handling.** Age was stored as text in the raw file, so the notebook label-encoded it as a category instead of treating it as a number. The API receives Age as a number, which does not match any of the encoder's text classes, so every request falls back to the same placeholder value. Cleaning Age into a numeric column and retraining would fix this.
- **Question the BMI feature.** Because the labels are largely defined by BMI, including it makes the task much easier. Training a model without Height, Weight and BMI would show how much the lifestyle features really contribute.
- **Use a scikit-learn `Pipeline`.** Bundling imputation, encoding, scaling and the model into one object would remove the need to keep four separate pickle files in sync, and would make the API simpler.
- **Use one-hot encoding for nominal columns.** Label encoding gives categories like MTRANS an artificial order. This matters less for tree-based models but is still not ideal.
- **Return probabilities.** Showing the model's confidence for each class would help a clinic judge borderline cases.
- **Pin library versions.** The pickle files were saved with one version of scikit-learn and can warn or break under another.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · FastAPI · Pydantic · Streamlit · Matplotlib · Seaborn

## Context

Built as the final exam project for the Model Deployment course, Data Science program, Bina Nusantara University, June 2025.

Alessandro Moreno Lawadinata
[LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN) · [GitHub](https://github.com/SeafoodEnthusiast)
