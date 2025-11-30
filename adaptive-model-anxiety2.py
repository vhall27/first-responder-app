import csv
import math

# -----------------------------
# File names (change if needed)
# -----------------------------
TRAIN_FILE = "csv-files/Mental-Illness-Train-Anxiety.csv"
TEST_FILE  = "csv-files/Mental-Illness-Test-Anxiety.csv"

# Exact column name for the 6-month duration question in your CSV
SIX_MONTH_COL = "Have the symptoms you answered yes to been present for more days than not for the past six months?"


#############################################
# 1. Load training and test data
#############################################

def load_training_data(filename):
    with open(filename, mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Filter out malformed/blank rows
        rows = [r for r in reader if len(r) == len(header)]
    feature_names = header[:-1]   # all columns except label
    label_index = len(header) - 1
    return feature_names, label_index, rows


def load_test_data(filename):
    with open(filename, mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if len(r) == len(header)]
    feature_names = header[:-1]
    label_index = len(header) - 1
    return feature_names, label_index, rows


#############################################
# 2. Compute item likelihoods P(x|y)
#############################################

def compute_item_likelihoods(rows, label_index, laplace=1.0):
    """
    rows: list of CSV rows (strings)
    label_index: index of label column (0/1 anxiety)
    laplace: Laplace smoothing factor

    Returns:
      p1[j] = P(feature_j = 1 | anxiety = 1)  (softened)
      p0[j] = P(feature_j = 1 | anxiety = 0)  (softened)
      prior = fixed prior P(anxiety = 1)
    """
    n = len(rows)
    pos_rows = [r for r in rows if int(r[label_index]) == 1]
    neg_rows = [r for r in rows if int(r[label_index]) == 0]

    n_pos = len(pos_rows)
    n_neg = len(neg_rows)

    # PRIOR: fixed from external knowledge (e.g., NIH / literature), happens to be the same as pos_rows/total rows
    prior = 0.40

    num_features = label_index  # all columns before label
    p1 = []
    p0 = []

    for j in range(num_features):
        pos_ones = sum(int(r[j]) for r in pos_rows)
        neg_ones = sum(int(r[j]) for r in neg_rows)

        # Basic Laplace-smoothed estimates
        raw_p1 = (pos_ones + laplace) / (n_pos + 2 * laplace) if n_pos > 0 else 0.5
        raw_p0 = (neg_ones + laplace) / (n_neg + 2 * laplace) if n_neg > 0 else 0.5

        p1.append(raw_p1)
        p0.append(raw_p0)

    # GLOBAL SHRINK: make all items less extreme (toward 0.5)
    alpha = 0.7  # 0 < alpha < 1; smaller = softer, less extreme
    for j in range(num_features):
        p1[j] = alpha * p1[j] + (1 - alpha) * 0.5
        p0[j] = alpha * p0[j] + (1 - alpha) * 0.5

    return p1, p0, prior


#############################################
# 3. Entropy, Bayes updating, information gain
#############################################

def entropy(p):
    # edge case
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return - (p * math.log2(p) + (1 - p) * math.log2(1 - p))


def update_posterior(pi, p1_q, p0_q, response):
    """
    pi: current P(anxiety=1 | history)
    p1_q: P(x=1 | anxiety=1) for this question
    p0_q: P(x=1 | anxiety=0) for this question
    response: 0 or 1
    """
    if response == 1:
        L1 = p1_q
        L0 = p0_q
    else:
        L1 = 1 - p1_q
        L0 = 1 - p0_q

    num = L1 * pi
    den = num + L0 * (1 - pi)
    return num / den if den != 0 else pi


def expected_information_gain(pi, p1_q, p0_q):
    # predictive probability of response = 1
    p_r1 = pi * p1_q + (1 - pi) * p0_q

    post1 = update_posterior(pi, p1_q, p0_q, 1)
    post0 = update_posterior(pi, p1_q, p0_q, 0)

    exp_H = p_r1 * entropy(post1) + (1 - p_r1) * entropy(post0)
    return entropy(pi) - exp_H


#############################################
# 4. Risk category labeling (5 levels)
#############################################

def label_category(pi):
    """
    Map final posterior probability to 5 categories:
      Very Low, Low, Moderate, High, Very High
    Tuned so mid-range appears more often.
    """
    if pi >= 0.43:
        return "High possibility of Anxiety Disorder, Consult a Physician"    
    elif pi <= 0.33:
        return "Low possibility of Anxiety Disorder"
    else:
        return "Moderate possibility of Anxiety Disorder, Consult a Physician"

#############################################
# 5. Interactive adaptive anxiety session
#############################################

def run_adaptive_anxiety_session():
    feature_names, label_index, rows = load_training_data(TRAIN_FILE)
    p1, p0, prior = compute_item_likelihoods(rows, label_index)

    # EXTRA SHRINK just for 6-month duration so it doesn't dominate
    if SIX_MONTH_COL in feature_names:
        target_idx = feature_names.index(SIX_MONTH_COL)
        special_alpha = 0.4  # stronger pull toward 0.5
        p1[target_idx] = special_alpha * p1[target_idx] + (1 - special_alpha) * 0.5
        p0[target_idx] = special_alpha * p0[target_idx] + (1 - special_alpha) * 0.5
    else:
        # This helps you debug if the name doesn't match
        print(f"WARNING: 6-month column '{SIX_MONTH_COL}' not found in feature_names.")

    print(f"Using prior P(anxiety=1) = {prior:.3f}")

    pi = prior
    asked = set()
    at_least_one_yes = False

    max_questions = len(feature_names)
    min_questions = 3            # must ask at least 3 questions
    high_conf = 0.90             # more conservative thresholds
    low_conf  = 0.10

    question_history = []

    for step in range(max_questions):

        # Stopping rule only after min_questions
        if step >= min_questions:
            if pi >= high_conf or pi <= low_conf:
                break

        # Choose next question via information gain
        best_q = None
        best_gain = -1

        for q_idx in range(len(feature_names)):
            if q_idx in asked:
                continue

            q_name = feature_names[q_idx]

            # Rule 1: 6-month question CANNOT be question 1 or 2:
            if q_name == SIX_MONTH_COL and step < 4:
                continue

            # Rule 2: Only ask 6-month if at least one YES
            if q_name == SIX_MONTH_COL and not at_least_one_yes:
                continue

            gain = expected_information_gain(pi, p1[q_idx], p0[q_idx])
            if gain > best_gain:
                best_gain = gain
                best_q = q_idx

        if best_q is None:
            break

        q_name = feature_names[best_q]

        # Ask the user
        while True:
            ans = input(f"Q{step+1}: {q_name} (0 = No, 1 = Yes): ").strip()
            if ans in ("0", "1"):
                ans = int(ans)
                break
            else:
                print("Please enter 0 or 1.")

        asked.add(best_q)

        if ans == 1:
            at_least_one_yes = True

        old_pi = pi
        pi = update_posterior(pi, p1[best_q], p0[best_q], ans)
        question_history.append((q_name, ans, pi))

        print(f"Updated probability: {old_pi:.3f} → {pi:.3f}\n")

    # Summary
    print("\n===== ADAPTIVE SESSION SUMMARY =====")
    for qname, ans, post in question_history:
        print(f"{qname}: answer={ans}, posterior={post:.3f}")

    print(f"\nFinal probability of anxiety = {pi:.3f}")

    # 5-category label
    print("\n===== FINAL ASSESSMENT =====")
    print(label_category(pi))


#############################################
# 6. Simulator for the adaptive model
#############################################

def simulate_adaptive_anxiety():
    # Load training data & likelihoods
    train_feature_names, train_label_index, train_rows = load_training_data(TRAIN_FILE)
    p1, p0, prior = compute_item_likelihoods(train_rows, train_label_index)

    # Shrinking the weight placed on the 6-month item in simulation too
    if SIX_MONTH_COL in train_feature_names:
        target_idx = train_feature_names.index(SIX_MONTH_COL)
        special_alpha = 0.4
        p1[target_idx] = special_alpha * p1[target_idx] + (1 - special_alpha) * 0.5
        p0[target_idx] = special_alpha * p0[target_idx] + (1 - special_alpha) * 0.5
    else:
        print(f"WARNING: 6-month column '{SIX_MONTH_COL}' not found in train_feature_names.")

    # Load test data
    test_feature_names, test_label_index, test_rows = load_test_data(TEST_FILE)

    # Ensure features align
    assert train_feature_names == test_feature_names, "Train/test feature order mismatch!"

    total = len(test_rows)
    tp = fp = tn = fn = 0
    question_counts = []

    min_questions = 3
    high_conf = 0.90
    low_conf  = 0.10

    # For evaluation criteria, treat Moderate+ as screen-positive:
    # P >= 0.43 => positive
    eval_threshold = 0.43

    for row in test_rows:
        true_label = int(row[test_label_index])
        answers = [int(x) for x in row[:-1]]  # simulated user responses

        pi = prior
        asked = set()
        at_least_one_yes = False
        num_questions = 0

        # Adaptive loop
        for step in range(len(train_feature_names)):

            # stopping rule (after min questions)
            if step >= min_questions:
                if pi >= high_conf or pi <= low_conf:
                    break

            best_q = None
            best_gain = -1

            for q_idx in range(len(train_feature_names)):
                if q_idx in asked:
                    continue

                q_name = train_feature_names[q_idx]

                # Rule 1: 6-month question cannot be Q1 or Q2
                if q_name == SIX_MONTH_COL and step < 2:
                    continue

                # Rule 2: only ask 6-month if at least one YES
                if q_name == SIX_MONTH_COL and not at_least_one_yes:
                    continue

                gain = expected_information_gain(pi, p1[q_idx], p0[q_idx])
                if gain > best_gain:
                    best_gain = gain
                    best_q = q_idx

            if best_q is None:
                break

            ans = answers[best_q]
            num_questions += 1
            asked.add(best_q)

            if ans == 1:
                at_least_one_yes = True

            pi = update_posterior(pi, p1[best_q], p0[best_q], ans)

        # Classification for metrics: Moderate+ = screen-positive
        pred = 1 if pi >= eval_threshold else 0

        if pred == 1 and true_label == 1: tp += 1
        if pred == 1 and true_label == 0: fp += 1
        if pred == 0 and true_label == 0: tn += 1
        if pred == 0 and true_label == 1: fn += 1

        question_counts.append(num_questions)

    # Metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    avg_questions = sum(question_counts) / total if total > 0 else 0.0

    print("\n===== ADAPTIVE ANXIETY SIMULATION RESULTS =====")
    print(f"Total test cases: {total}")
    print(f"Accuracy:      {accuracy:.3f}")
    print(f"Sensitivity:   {sensitivity:.3f}")
    print(f"Specificity:   {specificity:.3f}")
    print(f"Precision:     {precision:.3f}")
    print(f"Avg Questions: {avg_questions:.2f}")
    print(f"TP: {tp}   FP: {fp}   TN: {tn}   FN: {fn}")


#############################################
# 7. Main
#############################################

def main():
    # 1) Run an interactive session with a real user:
    # run_adaptive_anxiety_session()

    # 2) Or run the simulator:
    simulate_adaptive_anxiety()


if __name__ == "__main__":
    main()
