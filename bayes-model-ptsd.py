import csv
import math
import matplotlib.pyplot as plt

# -----------------------------
# File names (change if needed)
# -----------------------------
TRAIN_FILE = "csv-files/Mental-Illness-Train-PTSD.csv"
TEST_FILE  = "csv-files/Mental-Illness-Test-PTSD.csv"



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
    label_index: index of label column (0/1 ptsd)
    laplace: Laplace smoothing factor

    Returns:
      p1[j] = P(feature_j = 1 | ptsd = 1)  (softened)
      p0[j] = P(feature_j = 1 | ptsd = 0)  (softened)
      prior = fixed prior P(ptsd = 1)
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

def naive_bayes_posterior_for_row(answers, p1, p0, prior):
    """
    answers: list of 0/1 feature values for one person
    p1, p0: likelihood parameters from compute_item_likelihoods
    prior: P(ptsd=1)

    Returns: posterior P(ptsd=1 | answers) using Naive Bayes in log-odds space.
    """
    # Avoid extreme 0/1 just in case
    eps = 1e-9

    # log prior odds
    prior = min(max(prior, eps), 1.0 - eps)
    log_odds = math.log(prior / (1.0 - prior))

    for j, x in enumerate(answers):
        pj1 = min(max(p1[j], eps), 1.0 - eps)
        pj0 = min(max(p0[j], eps), 1.0 - eps)

        if x == 1:
            log_odds += math.log(pj1) - math.log(pj0)
        else:
            log_odds += math.log(1.0 - pj1) - math.log(1.0 - pj0)

    # convert log-odds back to probability
    prob = 1.0 / (1.0 + math.exp(-log_odds))
    return prob

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
    pi: current P(ptsd=1 | history)
    p1_q: P(x=1 | ptsd=1) for this question
    p0_q: P(x=1 | ptsd=0) for this question
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
# 4. Risk category labeling (3 levels)
#############################################

def label_category(pi):
    """
    Map final posterior probability to 5 categories:
      Very Low, Low, Moderate, High, Very High
    Tuned so mid-range appears more often.
    """
    if pi >= 0.35:
        return "High possibility of PTSD, Consult a Physician"    
    elif pi <= 0.20:
        return "Low possibility of PTSD"
    else:
        return "Moderate possibility of PTSD, Consult a Physician"

#############################################
# 5. Interactive adaptive ptsd session
#############################################

def run_adaptive_ptsd_session():
    feature_names, label_index, rows = load_training_data(TRAIN_FILE)
    feature_names_test, label_index_test, rows_test = load_training_data(TEST_FILE)
    p1, p0, prior = compute_item_likelihoods(rows, label_index)

    print(f"Using prior P(ptsd=1) = {prior:.3f}")

    pi = prior
    asked = set()
    at_least_one_yes = False

    max_questions = len(feature_names)
    min_questions = 3            # must ask at least 3 questions
    high_conf = 0.80             # more conservative thresholds
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
            
            # only test file has full question labels instead of general categories for inputs
            q_name = feature_names_test[q_idx]

            gain = expected_information_gain(pi, p1[q_idx], p0[q_idx])
            if gain > best_gain:
                best_gain = gain
                best_q = q_idx

        if best_q is None:
            break

        q_name = feature_names_test[best_q]

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

    print(f"\nFinal probability of ptsd = {pi:.3f}")

    # 5-category label
    print("\n===== FINAL ASSESSMENT =====")
    print(label_category(pi))


#############################################
# 6. Simulator for the adaptive model
#############################################

def simulate_adaptive_ptsd():
    # Load training data & likelihoods
    train_feature_names, train_label_index, train_rows = load_training_data(TRAIN_FILE)
    p1, p0, prior = compute_item_likelihoods(train_rows, train_label_index)

    # Load test data
    test_feature_names, test_label_index, test_rows = load_test_data(TEST_FILE)

    total = len(test_rows)
    tp = fp = tn = fn = 0
    question_counts = []

    min_questions = 3
    high_conf = 0.90
    low_conf  = 0.10

    # For evaluation criteria, treat Moderate+ as screen-positive:
    # P >= 0.5 => positive
    eval_threshold = 0.5

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

                q_name = test_feature_names[q_idx]

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

    print("\n===== ADAPTIVE PTSD SURVEY SIMULATION RESULTS =====")
    print(f"Total test cases: {total}")
    print(f"Accuracy:      {accuracy:.3f}")
    print(f"Sensitivity:   {sensitivity:.3f}")
    print(f"Specificity:   {specificity:.3f}")
    print(f"Precision:     {precision:.3f}")
    print(f"Avg Questions: {avg_questions:.2f}")
    print(f"TP: {tp}   FP: {fp}   TN: {tn}   FN: {fn}")

def visualize_naive_bayes_calibration(num_bins=10):
    # 1) Fit NB on training data
    train_feature_names, train_label_index, train_rows = load_training_data(TRAIN_FILE)
    p1, p0, prior = compute_item_likelihoods(train_rows, train_label_index)

    # 2) Load test data
    test_feature_names, test_label_index, test_rows = load_test_data(TEST_FILE)

    y_true = []
    y_prob = []

    for row in test_rows:
        label = int(row[test_label_index])
        answers = [int(x) for x in row[:-1]]

        prob = naive_bayes_posterior_for_row(answers, p1, p0, prior)
        y_true.append(label)
        y_prob.append(prob)

    # 3) Bin predictions
    bin_counts = [0] * num_bins
    bin_pos = [0] * num_bins
    bin_sum_pred = [0.0] * num_bins

    for yt, yp in zip(y_true, y_prob):
        # put prob in a bin [0,1) split into num_bins
        idx = int(yp * num_bins)
        if idx == num_bins:  # handle yp == 1.0
            idx = num_bins - 1
        bin_counts[idx] += 1
        bin_pos[idx] += yt
        bin_sum_pred[idx] += yp

    bin_mean_pred = []
    bin_frac_pos = []

    for i in range(num_bins):
        if bin_counts[i] > 0:
            mean_pred = bin_sum_pred[i] / bin_counts[i]
            frac_pos = bin_pos[i] / bin_counts[i]
            bin_mean_pred.append(mean_pred)
            bin_frac_pos.append(frac_pos)

    # 4) Plot calibration curve
    plt.figure()
    # ideal calibration line
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="Perfect calibration")
    # NB empirical calibration
    plt.plot(bin_mean_pred, bin_frac_pos, marker="o", label="Naive Bayes")

    plt.xlabel("Predicted probability (Naive Bayes)")
    plt.ylabel("Empirical fraction with ptsd")
    plt.title("Calibration curve – Naive Bayes (ptsd)")
    plt.legend()
    plt.grid(True)

    # 5) Optional: histogram of predicted probabilities
    plt.figure()
    plt.hist(y_prob, bins=num_bins)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Distribution of Naive Bayes predicted probabilities")

    plt.show()

#############################################
# 7. Main
#############################################

def main():
    # 1) Run an interactive session with a real user:
    # run_adaptive_ptsd_session()

    # 2) Or run the simulator:
    simulate_adaptive_ptsd()

    visualize_naive_bayes_calibration()


if __name__ == "__main__":
    main()
