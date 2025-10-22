import numpy as np
import json

# Load CART rules
#change to "cart_rules_with_prediction.json" file pathname
with open("/Users/javier/cart_rules_with_prediction.json", "r") as f:
    cart_rules_raw = json.load(f)

# Convert to usable format
cart_rules = {}
for key, val in cart_rules_raw.items():
    cond = val["conditions"][0]  # get string from list
    pred = val["prediction"][0]  # get int from list
    # Map R factor to 0/1 (assuming 1 = No, 2 = Yes in R)
    if pred > 1:
        pred = 1
    cart_rules[key] = {
        "conditions": cond,
        "prediction": pred
    }

# Function to evaluate a single rule
def eval_rule(rule_str, inputs):
    if rule_str.strip() == "":  # empty rule
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

# CART predictor
def cart_predict(inputs):
    for key in sorted(cart_rules.keys(), key=lambda x: int(x)):
        rule = cart_rules[key]
        if eval_rule(rule["conditions"], inputs):
            return rule["prediction"]
    return 0

# ---------------------------
# Logistic regression coefficients
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
# Ask user for input
# ---------------------------
def get_user_input():
    print("Enter the following inputs:")
    data = {}
    data['HighBP'] = int(input("HighBP (0/1): "))
    data['HighChol'] = int(input("HighChol (0/1): "))
    data['BMI'] = float(input("BMI: "))
    data['HeartDiseaseorAttack'] = int(input("Heart Disease/Attack (0/1): "))
    data['PhysActivity'] = int(input("Physical Activity (0/1): "))
    data['HvyAlcoholConsump'] = int(input("Heavy Alcohol Consumption (0/1): "))
    data['GenHlth'] = float(input("General Health (1-5): "))
    data['DiffWalk'] = int(input("Difficulty Walking (0/1): "))
    data['Sex'] = int(input("Sex (1=Male, 0=Female): "))
    data['Age'] = float(input("Age: "))
    return data

# ---------------------------
# Run predictions
# ---------------------------
user_data = get_user_input()

log_prob = logistic_predict(user_data)
cart_pred = cart_predict(user_data)

print(f"\nLogistic Regression Probability: {log_prob:.4f}")
print(f"CART Prediction (0 = No Diabetes, 1 = Diabetes): {cart_pred}")
