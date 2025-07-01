project_summary = """
# ✅ Project: Healthcare Cost Prediction using Random Forest

---

## 📥 Input Columns (Features):

- **Age** – Patient's age  
- **BMI** – Body Mass Index  
- **SmokingStatus** – Smoker or Non-Smoker (converted to 0/1)  
- **ChronicCondition** – Yes or No (converted to 0/1)  

---

## 🎯 Output Column (Target):

- **AnnualCost** – Yearly healthcare expense (₹)  
*(This is what we are predicting)*

---

## 🤖 How the Model Predicts:

- **Model Used**: ✅ RandomForestRegressor  
- Trained with input features to learn patterns in medical cost  
- Checks how age, BMI, smoking, and chronic illness affect the cost  
- For new input data, it predicts the expected **Annual Medical Cost**

---

## 🧪 Model Evaluation Results:

- ✅ **MAE (Mean Absolute Error)**: ₹206.90  
- ✅ **R² Score**: 0.98 (98% accuracy)

---

## 💡 Real Use Case:

Insurance companies can use this model to estimate a person's medical cost based on their health profile.  
It helps in setting fair premium amounts and identifying high-risk customers.

---
