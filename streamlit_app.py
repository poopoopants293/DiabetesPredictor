import streamlit as st
import numpy as np
import json
import os

# ---------------------------
# Load CART rules (relative path)
# ---------------------------
file_path = os.path.join(os.path.dirname(__file__), "cart_rules_with_prediction.json")
with open(file_path, "r") as f:
    cart_rules_raw = json.load(f)

# Convert to usable format
cart_rules = {}
for key, val in cart_rules_raw.items():
    cond = val["conditions"][0]  # get string from list
    pred = val["prediction"][0]  # get int from list
    if pred > 1:
        pred = 1
    cart_rules[key] = {
        "conditions": cond,
        "prediction": pred
    }

# ---------------------------
# CART functions
# ---------------------------
def eval_rule(rule_str, inputs):
    if rule_str.strip() == "":
        return False
    conditions = rule_str.split("&")[1:]  # skip "root"
    for cond in conditions:
        cond = cond.strip()
        if ">=" in cond:
            var, val = cond.split(">=")
            if inputs[var.strip()] < float(val.strip()):
                return False
        elif "<" in cond:
            var, val = cond.split("<")
            if inputs[var.strip()] >= float(val.strip()):
                return False
        elif "=" in cond:
            var, val = cond.split("=")
            if inputs[var.strip()] != int(val.strip()):
                return False
    return True

def cart_predict(inputs):
    for key in sorted(cart_rules.keys(), key=lambda x: int(x)):
        rule = cart_rules[key]
        if eval_rule(rule["conditions"], inputs):
            return rule["prediction"]
    return 0

# ---------------------------
# Logistic regression
# ---------------------------
coeffs = {
    '(Intercept)': -5.597316,
    'HighBP': 0.736536,
    'HighChol': 0.576521,
    'BMI': 0.070961,
    'HeartDiseaseorAttack': 0.305751,
    'PhysActivity': -0.237292,
    'HvyAlcoholConsump': -0.762182,
    'GenHlth': 0.528098,
    'DiffWalk': 0.135391,
    'Sex': 0.267340,
    'Age': 0.151141
}

def logistic_predict(inputs):
    logit = coeffs['(Intercept)']
    for var, value in inputs.items():
        if var in coeffs:
            logit += coeffs[var] * value
    prob = 1 / (1 + np.exp(-logit))
    return prob

# ---------------------------
# Streamlit Interface
# ---------------------------
st.title("Diabetes Prediction App")
st.markdown("Enter patient data to get predictions from both Logistic Regression and CART model.")

# Sidebar inputs
st.sidebar.header("Patient Inputs")
HighBP = st.sidebar.radio("HighBP", [0, 1])
HighChol = st.sidebar.radio("HighChol", [0, 1])
BMI = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
HeartDiseaseorAttack = st.sidebar.radio("Heart Disease/Attack", [0, 1])
PhysActivity = st.sidebar.radio("Physical Activity", [0, 1])
HvyAlcoholConsump = st.sidebar.radio("Heavy Alcohol Consumption", [0, 1])
GenHlth = st.sidebar.slider("General Health (1=Poor, 5=Excellent)", 1.0, 5.0, 3.0)
DiffWalk = st.sidebar.radio("Difficulty Walking", [0, 1])
Sex = st.sidebar.radio("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
Age = st.sidebar.slider("Age", 0, 120, 30)

# Collect inputs into dictionary
user_data = {
    "HighBP": HighBP,
    "HighChol": HighChol,
    "BMI": BMI,
    "HeartDiseaseorAttack": HeartDiseaseorAttack,
    "PhysActivity": PhysActivity,
    "HvyAlcoholConsump": HvyAlcoholConsump,
    "GenHlth": GenHlth,
    "DiffWalk": DiffWalk,
    "Sex": Sex,
    "Age": Age
}

# Predict button
if st.button("Predict"):
    log_prob = logistic_predict(user_data)
    cart_pred = cart_predict(user_data)

    st.subheader("Prediction Results")
    st.write(f"**Logistic Regression Probability of Diabetes:** {log_prob:.4f}")
    st.write(f"**CART Prediction:** {'Diabetes (1)' if cart_pred==1 else 'No Diabetes (0)'}")
