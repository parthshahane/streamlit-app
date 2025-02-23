import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
pip install scikit-learn


st.title("Titanic Survival Prediction")
scaler = StandardScaler()
scaler.fit(X)

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ['Male', 'Female'])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 32.0)
embarked = st.selectbox("Embarked", ['C', 'Q', 'S'])

sex = 1 if sex == 'Male' else 0
embarked = {'C': 0, 'Q': 1, 'S': 2}[embarked]
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
input_data = scaler.transform(input_data)

if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    result = "Survived" if prediction[0] == 1 else "Did not survive"
    st.write(f"Prediction: {result}")
