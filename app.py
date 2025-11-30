from flask import Flask, render_template, request, redirect, url_for
import math
import csv
import json

app = Flask(__name__)

# Config for each screening type
TEST_CONFIGS = {
    "depression": {
        "label": "Major Depressive Disorder",
        "test_file": "csv-files/Mental-Illness-Test-Depression.csv",
        "theta_file": "theta_depression.json",
        "description": "Screen for symptoms of depression tailored to first responders.",
    },
    "anxiety": {
        "label": "Generalized Anxiety Disorder",
        "test_file": "csv-files/Mental-Illness-Test-Anxiety.csv",
        "theta_file": "theta_anxiety.json",
        "description": "Screen for operational and chronic anxiety in first responders.",
    },
    "ptsd": {
        "label": "PTSD",
        "test_file": "csv-files/Mental-Illness-Test-PTSD.csv",
        "theta_file": "theta_ptsd.json",
        "description": "Screen for post-traumatic stress symptoms after critical incidents.",
    },
}


def dot_prod(theta_list, example):
    """Dot product of theta and example, including bias term."""
    total = theta_list[0]  # bias
    for i in range(1, len(theta_list)):
        total += theta_list[i] * float(example[i - 1])
    return total


def load_questions(test_file):
    """Get the header row and return all feature columns except label."""
    with open(test_file, mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return header[:-1]


def classify(prob, test_type):
    """
    Convert probability into a severity label, using
    calibration-specific thresholds for each disorder.
    """

    # Depression thresholds, based on calibration data
    # Low:        p < 0.50
    # Moderate:   0.50 ≤ p < 0.60
    # High:       0.60 ≤ p < 0.70
    # Very High:  p ≥ 0.70
    if test_type == "depression":
        if prob >= 0.70:
            return "Very High Possibility"
        elif prob >= 0.60:
            return "High Possibility"
        elif prob >= 0.50:
            return "Moderate Possibility"
        else:
            return "Low Possibility"

    # Anxiety thresholds, based on calibration data
    # Low:        p < 0.65
    # Moderate:   0.65 ≤ p < 0.80
    # High:       0.80 ≤ p < 0.90
    # Very High:  p ≥ 0.90
    elif test_type == "anxiety":
        if prob >= 0.90:
            return "Very High Possibility"
        elif prob >= 0.80:
            return "High Possibility"
        elif prob >= 0.65:
            return "Moderate Possibility"
        else:
            return "Low Possibility"

    # PTSD thresholds, based on calibration data
    # Low:        p < 0.40
    # Moderate:   0.40 ≤ p < 0.60
    # High:       0.60 ≤ p < 0.80
    # Very High:  p ≥ 0.80
    elif test_type == "ptsd":
        if prob >= 0.80:
            return "Very High Possibility"
        elif prob >= 0.60:
            return "High Possibility"
        elif prob >= 0.40:
            return "Moderate Possibility"
        else:
            return "Low Possibility"

    # Fallback (shouldn't really be hit)
    else:
        if prob >= 0.85:
            return "Very High Possibility"
        elif prob >= 0.60:
            return "High Possibility"
        elif prob >= 0.40:
            return "Moderate Possibility"
        else:
            return "Low Possibility"



# Loads all models + questions once
MODELS = {}

for key, cfg in TEST_CONFIGS.items():
    with open(cfg["theta_file"], "r") as f:
        thetas = json.load(f)
    questions = load_questions(cfg["test_file"])
    MODELS[key] = {
        "label": cfg["label"],
        "theta": thetas,
        "questions": questions,
        "description": cfg["description"],
    }


# Routes Below

@app.route("/")
def home():
    # Landing page (choose which test)
    return render_template("home.html", test_configs=TEST_CONFIGS)


@app.route("/screen/<test_type>", methods=["GET", "POST"])
# The different screenings themselves
def screen(test_type):
    if test_type not in MODELS:
        return redirect(url_for("home"))

    model = MODELS[test_type]
    questions = model["questions"]

    result_text = None
    probability = None
    answers = {}
    summary_answers = None

    if request.method == "POST" and request.form.get("submit_btn") == "run_test":
        user_answers = []

        for i, _ in enumerate(questions):
            key = f"q{i}"
            val = request.form.get(key, "0")
            if val not in ("0", "1"):
                val = "0"
            user_answers.append(float(val))
            answers[key] = val

        # logistic regression prediction
        z = dot_prod(model["theta"], user_answers)
        sig = 1 / (1 + math.exp(-z))
        probability = round(sig, 3)

        severity = classify(sig, test_type)
        result_text = f"{severity} of {model['label']}"

        # Build summary of answers for display
        summary_answers = []
        for i, q in enumerate(questions):
            key = f"q{i}"
            val = answers.get(key, "0")
            summary_answers.append({
                "question": q,
                "answer": "Yes" if val == "1" else "No"
            })

    return render_template(
        "screen.html",
        test_key=test_type,
        test_label=model["label"],
        test_description=model["description"],
        questions=questions,
        result_text=result_text,
        probability=probability,
        answers=answers,
        summary_answers=summary_answers,
    )


if __name__ == "__main__":
    app.run(debug=True)
