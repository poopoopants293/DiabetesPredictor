import streamlit as st
import numpy as np
import json
import os
####trial
import openai
import streamlit as st

openai.api_key = DiabetesPredictor.secrets["OPENAI_API_KEY"]
#from dotenv import load_dotenv
#load_dotenv()  # Loads variables from .env file
#openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Load CART rules (relative path)
# ---------------------------
file_path = os.path.join(os.path.dirname(__file__), "cart")
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
age_bins = np.linspace(18, 80, 14)
def age_to_factor(age):
    # Ensure age is within bounds
    age = min(max(age, 18), 80)
    # digitize returns bin index (1-based)
    factor = np.digitize(age, age_bins, right=False)
    # Clip to 1–13
    factor = min(factor, 13)
    return factor
# ---------------------------
# Streamlit Interface
# ---------------------------
st.title("Diabetes Prediction App")
st.markdown("Enter data to get prediction.")

def generate_advice(prediction, probability):
    """
    Generate a tailored health message based on the predicted diabetes status 
    and estimated probability. Uses rules for risk, then calls OpenAI to rephrase 
    the message naturally in a friendly tone.
    """
    # 1️⃣ Base message based on prediction and probability
    if prediction == 1:
        base_message = (
            "You are predicted to be at high risk of diabetes. "
            "It is strongly recommended to consult a healthcare professional. "
            "Monitor your blood sugar, maintain a healthy diet, and exercise regularly."
        )
    elif probability > 0.7:
        base_message = (
            f"Your prediction indicates no diabetes, but your estimated probability is {probability:.2f}. "
            "This suggests a high chance of developing diabetes. Please consider consulting a healthcare professional. "
            "Maintain a healthy lifestyle and monitor your health closely."
        )
    else:
        base_message = (
            f"Your estimated probability of developing diabetes is {probability:.2f}, and your prediction shows low risk. "
            "You are currently at low risk, but prevention is important. "
            "Maintain a balanced diet, stay active, and monitor your health regularly."
        )

    # 2️⃣ Ask OpenAI to rephrase in a friendly and supportive tone
    prompt = (
        "Rephrase the following health advice in a friendly, clear, and supportive tone, and add additional recommendations based on individual's likelihood"
        "without mentioning specific models or technical terms:\n\n"
        f"{base_message}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful and supportive health assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.7
        )
        advice = response.choices[0].message.content.strip()
    except Exception as e:
        # If OpenAI fails, fallback to base message
        advice = base_message
        print(f"OpenAI API call failed: {e}")

    return advice

# Sidebar inputs
st.sidebar.header("Patient Inputs")
HighBP = st.sidebar.radio("High Blood Pressure (Yes(1), No(0))", [0, 1])
HighChol = st.sidebar.radio("High Cholesterol (Yes(1), No(0))", [0, 1])
BMI = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
HeartDiseaseorAttack = st.sidebar.radio("Heart Disease/Attack before? (Yes(1), No(0))", [0, 1])
PhysActivity = st.sidebar.radio("Physical activity in past 30 days? (Yes(1), No(0))", [0, 1])
HvyAlcoholConsump = st.sidebar.radio("Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) (Yes(1), No(0))", [0, 1])
GenHlth = st.sidebar.slider("General Health (5=Poor, 1=Excellent)", 1.0, 5.0, 3.0)
DiffWalk = st.sidebar.radio("Difficulty Walking (Yes(1), No(0))", [0, 1])
Sex = st.sidebar.radio("Sex (Male(1), Female(0))", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
Age = st.sidebar.slider("Age", 18, 80, 30)
Age_factor = age_to_factor(Age)

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
    "Age": Age_factor
}

# Predict button
if st.button("Predict"):
    log_prob = logistic_predict(user_data)
    cart_pred = cart_predict(user_data)

    st.subheader("Prediction Results")
    st.write(f"**Estimated probability of developing of Diabetes:** {log_prob:.4f}")
    st.write(f"**Predicted Diabetes Status (1 = High Risk!):** {'Diabetes (1)' if cart_pred==1 else 'No Diabetes (0)'}")
    advice = generate_advice(cart_pred, log_prob)
    st.subheader("Personalized Health Advice")
    st.write(advice)

