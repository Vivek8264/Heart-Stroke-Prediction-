import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==================================================
# PART 1: HEART STROKE PREDICTION (ML MODEL)
# ==================================================

# Load Dataset
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Handle Missing Values
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

# Encode Categorical Columns
le = LabelEncoder()
categorical_cols = ['gender', 'ever_married', 'work_type', 'smoking_status']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Feature Selection
X = data[['age','gender','hypertension','heart_disease',
          'bmi','avg_glucose_level','smoking_status',
          'work_type','ever_married']]
y = data['stroke']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("\nStroke Model Accuracy:", accuracy_score(y_test, y_pred))

# ==================================================
# STROKE USER INPUT
# ==================================================

print("\n----- ENTER STROKE RELATED DETAILS -----")

age = float(input("Age: "))
gender = input("Gender (Male/Female): ")
hypertension = int(input("Hypertension (1=Yes, 0=No): "))
heart_disease = int(input("Heart Disease (1=Yes, 0=No): "))
bmi = float(input("BMI: "))
avg_glucose = float(input("Average Glucose Level: "))
smoking = input("Smoking (never/formerly/smokes): ")
work = input("Work Type (Private/Govt/self-employed): ")
married = input("Married (Yes/No): ")

# Encode Input
gender = 1 if gender.lower() == "male" else 0
married = 1 if married.lower() == "yes" else 0

smoking_dict = {"never":0, "formerly":1, "smokes":2}
work_dict = {"private":0, "govt":1, "self-employed":2}

smoking = smoking_dict.get(smoking.lower(), 0)
work = work_dict.get(work.lower(), 0)

# Prediction
user_data = np.array([[age, gender, hypertension, heart_disease,
                       bmi, avg_glucose, smoking, work, married]])
user_data = scaler.transform(user_data)
stroke_pred = model.predict(user_data)

print("\n----- STROKE PREDICTION -----")
if stroke_pred[0] == 1:
    print("⚠️ HIGH RISK OF STROKE")
else:
    print("✅ LOW RISK OF STROKE")

# ==================================================
# PART 2: HEART ATTACK PREDICTION (RULE-BASED)
# ==================================================

print("\n----- ENTER HEART ATTACK RELATED DETAILS -----")

age_ha = int(input("Age: "))
gender_ha = input("Gender (Male/Female): ")
bp_sys = float(input("Systolic BP: "))
bp_dia = float(input("Diastolic BP: "))
cholesterol = float(input("Total Cholesterol: "))
blood_sugar = float(input("Fasting Blood Sugar: "))
bmi_ha = float(input("BMI: "))
smoking_ha = input("Smoking (non/former/current): ")
activity = input("Physical Activity (active/moderate/sedentary): ")
family_history = int(input("Family History (1=Yes, 0=No): "))
stress = int(input("Chronic Stress (1=Yes, 0=No): "))
chest_pain = int(input("Chest Pain (1=Yes, 0=No): "))
breath_short = int(input("Shortness of Breath (1=Yes, 0=No): "))
cold_sweat = int(input("Cold Sweat (1=Yes, 0=No): "))
nausea = int(input("Nausea/Dizziness (1=Yes, 0=No): "))

# Risk Calculation
risk_score = 0

if age_ha > 60: risk_score += 2
elif age_ha >= 40: risk_score += 1

if gender_ha.lower() == "male": risk_score += 1
if bp_sys >= 140 or bp_dia >= 90: risk_score += 2
elif bp_sys >= 120: risk_score += 1

if cholesterol >= 240: risk_score += 2
elif cholesterol >= 200: risk_score += 1

if blood_sugar >= 126: risk_score += 2
elif blood_sugar >= 100: risk_score += 1

if bmi_ha >= 30: risk_score += 2
elif bmi_ha >= 25: risk_score += 1

if smoking_ha.lower() == "current": risk_score += 2
elif smoking_ha.lower() == "former": risk_score += 1

if activity.lower() == "sedentary": risk_score += 2
elif activity.lower() == "moderate": risk_score += 1

if family_history == 1: risk_score += 2
if stress == 1: risk_score += 1

if chest_pain == 1: risk_score += 3
if breath_short == 1: risk_score += 2
if cold_sweat == 1: risk_score += 1
if nausea == 1: risk_score += 1

# Result
print("\n----- HEART ATTACK PREDICTION -----")

if risk_score >= 10:
    print("🚨 HIGH RISK OF HEART ATTACK")
elif risk_score >= 5:
    print("⚠️ MODERATE RISK OF HEART ATTACK")
else:
    print("✅ LOW RISK OF HEART ATTACK")

# ==================================================
# FINAL CONCLUSION
# ==================================================
print("\n===== FINAL CONCLUSION =====")
print("Stroke prediction is done using Machine Learning.")
print("Heart attack prediction is done using medical rule-based analysis.")
print("This system helps in early risk detection and preventive healthcare.")