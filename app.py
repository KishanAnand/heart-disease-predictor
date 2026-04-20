import streamlit as st
import numpy as np
import joblib

st.title("Heart Disease Prediction (Multiple Models)")

models = {
    "Logistic Regression": joblib.load('lr.pkl'),
    "Naive Bayes": joblib.load('nb.pkl'),
    "SVM": joblib.load('svm.pkl'),
    "KNN": joblib.load('knn.pkl'),
    "Decision Tree": joblib.load('dt.pkl'),
    "XGBoost": joblib.load('xgb.pkl'),
    
}
model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

st.write("### Enter Patient Details")

age = st.number_input("Age", 1, 120)
sex_option = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex_option == "Male" else 0
cp = st.selectbox("Chest Pain Type", [0,1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs_value = st.number_input("Fasting Blood Sugar (mg/dl)", min_value=0.0)

fbs = 1 if fbs_value > 120 else 0
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of vessels", [0, 1, 2, 3])
thal = st.selectbox("Thal", [3, 6, 7])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)

    st.subheader(f"Model Used: {model_name}")

    if prediction[0] == 1:
        st.error("⚠️ High chance of Heart Disease \nPlease consult a doctor for further evaluation.")
    else:
        st.success("✅ Low chance of Heart Disease")

    try:
        prob = model.predict_proba(input_data)[0][1]
        st.write(f"Risk Score: {prob:.2f}")
    except:
        st.info("Probability not available for this model")