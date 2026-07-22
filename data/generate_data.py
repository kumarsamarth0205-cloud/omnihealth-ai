"""
generate_data.py
----------------
Generates a realistic synthetic heart disease patient dataset (500 rows).
No external libraries needed — uses only Python's built-in 'random' and 'csv' modules.

Run this FIRST before training the model:
    python data/generate_data.py
"""

import csv
import random
import os

random.seed(42)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "patients.csv")

# --- Column definitions (based on UCI Heart Disease dataset structure) ---
# age: 29–77
# sex: 1=Male, 0=Female
# cp: chest pain type (0=typical angina, 1=atypical angina, 2=non-anginal, 3=asymptomatic)
# trestbps: resting blood pressure (94–200 mmHg)
# chol: serum cholesterol (126–564 mg/dl)
# fbs: fasting blood sugar > 120 mg/dl (1=true, 0=false)
# restecg: resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)
# thalach: max heart rate achieved (71–202 bpm)
# exang: exercise induced angina (1=yes, 0=no)
# oldpeak: ST depression induced by exercise (0.0–6.2)
# slope: slope of peak exercise ST segment (0=upsloping, 1=flat, 2=downsloping)
# ca: number of major vessels colored by fluoroscopy (0–3)
# thal: 1=normal, 2=fixed defect, 3=reversible defect
# target: 1=Disease Present, 0=No Disease

HEADERS = [
    "name", "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

MALE_NAMES = [
    "Arjun Sharma", "Rahul Verma", "Aakash Patel", "Vikram Singh", "Rohit Gupta",
    "Manish Yadav", "Suresh Kumar", "Deepak Mehta", "Nitin Joshi", "Amit Shah",
    "Karan Malhotra", "Rajan Nair", "Sanjay Tiwari", "Ankit Chauhan", "Varun Reddy",
    "Pradeep Mishra", "Harish Pillai", "Gaurav Bhatt", "Rajesh Iyer", "Naveen Bansal",
]
FEMALE_NAMES = [
    "Priya Sharma", "Neha Gupta", "Anjali Verma", "Sunita Patel", "Kavita Singh",
    "Rekha Yadav", "Swati Mehta", "Pooja Joshi", "Meena Shah", "Divya Nair",
    "Kritika Reddy", "Shweta Mishra", "Ritu Pillai", "Ananya Bhatt", "Pallavi Iyer",
    "Shruti Bansal", "Deepika Chauhan", "Aishwarya Tiwari", "Mansi Kumar", "Asha Malhotra",
]


def generate_patient(pid):
    sex = random.choice([0, 1])
    age = random.randint(29, 77)
    name_pool = MALE_NAMES if sex == 1 else FEMALE_NAMES
    name = random.choice(name_pool)

    # Higher risk if older, male, or asymptomatic chest pain
    cp = random.choices([0, 1, 2, 3], weights=[20, 20, 20, 40])[0]
    trestbps = random.randint(94, 200)
    chol = random.randint(126, 564)
    fbs = 1 if chol > 200 and random.random() > 0.6 else 0
    restecg = random.choices([0, 1, 2], weights=[50, 30, 20])[0]
    thalach = random.randint(71, 202)
    exang = random.choices([0, 1], weights=[60, 40])[0]
    oldpeak = round(random.uniform(0.0, 6.2), 1)
    slope = random.choices([0, 1, 2], weights=[30, 40, 30])[0]
    ca = random.choices([0, 1, 2, 3], weights=[50, 25, 15, 10])[0]
    thal = random.choices([1, 2, 3], weights=[40, 20, 40])[0]

    # Simple rule-based target: disease more likely with risk factors
    risk_score = 0
    if age > 55: risk_score += 2
    if sex == 1: risk_score += 1
    if cp == 3: risk_score += 2
    if trestbps > 140: risk_score += 1
    if chol > 240: risk_score += 1
    if exang == 1: risk_score += 2
    if oldpeak > 2.0: risk_score += 2
    if ca > 0: risk_score += 2
    if thal == 3: risk_score += 2
    if thalach < 130: risk_score += 1

    target = 1 if risk_score >= 7 else 0
    # Add some noise
    if random.random() < 0.1:
        target = 1 - target

    return [name, age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak, slope, ca, thal, target]


def main():
    rows = [generate_patient(i) for i in range(500)]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(rows)
    print(f"[OK] Dataset generated: {OUTPUT_FILE}  ({len(rows)} patients)")


if __name__ == "__main__":
    main()
