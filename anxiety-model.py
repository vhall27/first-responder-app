import math
import csv
import matplotlib.pyplot as plt


# takes dot product of the theta values by the x vector
def dot_prod(theta_list, example):
  total = theta_list[0]  # adding bias term to the dot product
  # starting from index 1 because we already added the bias term,
  # accounting for the lack of bias term in "example"
  for i in range(1, len(theta_list)):
    total += theta_list[i] * float(example[i - 1])
  return total


def logistic_regression_trainer():
  with open('csv-files/Mental-Illness-Train-Anxiety.csv', mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    examples = list(csv_reader)


  # number of elements in theta/gradient without label column
  m_elems = len(header) - 1
  elems_with_bias = m_elems + 1
  # thetas, which include theta_0
  theta_list = [0] * (elems_with_bias)

  # given to us
  step_size = 0.0001
  iterations = 1000


  for i in range(iterations):
    # initialize the gradients
    gradient_list = [0] * (elems_with_bias)

    # iterates through rows in the csv file list "examples"
    for example in examples:
      y = int(example[-1])
      sigmoid = 1 / (1 + math.exp(-dot_prod(theta_list, example)))
      second_term = y - sigmoid

      # gradient for the bias term, always multiplied by a constant 1
      gradient_list[0] += second_term

      # calculates gradients of the "regular" terms (after the bias term),
      # so we are skipping the bias gradient term
      for j in range(1, elems_with_bias):
        # still need the zeroth term from example because it lacks a bias term
        first_term = float(example[j - 1])
        gradient_list[j] += first_term * second_term

    # last step in finding the new thetas, multiply each theta by the step_size
    for j in range(len(theta_list)):
      theta_list[j] += step_size * gradient_list[j]

  return theta_list

# tests the logistic regression based on test data given
def logistic_regression_tester(theta_list):
  with open('csv-files/Mental-Illness-Test-Anxiety.csv', mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    tests = list(csv_reader)

    # initialize a test "answers" or predictions
    predictions = []
    # iterates through each row, which is a different test case with different thetas
    for test in tests:
      sigmoid = 1 / (1 + math.exp(-dot_prod(theta_list, test)))
      # gives a bernoulli prediction equal to either one or zero
      if sigmoid >= 0.5:
        predictions.append(1)
      else:
        predictions.append(0)
    return predictions, tests
    
def accuracy(predictions, tests):
    # prints the predictions against the true answer that was provided, and calculates accuracy
    correct = 0
    true_pos_detected = 0
    total_pos = 0
    false_pos = 0
    for i in range(len(predictions)):
      print(f"Prediction: {predictions[i]} Actual: {tests[i][-1]}")
      if predictions[i] == int(tests[i][-1]):
        correct += 1
      if int(tests[i][-1]) == 1:
        total_pos += 1
      if predictions[i] == 1 and int(tests[i][-1]) == 1:
        true_pos_detected += 1
      elif predictions[i] == 1 and int(tests[i][-1]) == 0:
        false_pos += 1
    # prints accuracy "grade"
    print(f"Accuracy: {correct / len(predictions)}")
    print(len(predictions))
    # prints the amount of positive cases detected against the amount of total positives and false positives
    print(f"Detected true positives: {true_pos_detected} Total true positives: {total_pos} False positives: {false_pos}")


def calibration_curve(theta_list, test_rows, n_bins=10):
    """
    Computes calibration curve data for reliability plotting.
    Returns:
        bin_centers: middle of each probability bin
        avg_predicted: average predicted probability in each bin
        avg_observed: actual outcome frequency in each bin
        probs, labels: raw predicted probabilities and true labels
    """
    probs = []
    labels = []
    for row in test_rows:
        y = int(row[-1])
        p = 1 / (1 + math.exp(-dot_prod(theta_list, row)))
        probs.append(p)
        labels.append(y)

    bins = [i / n_bins for i in range(n_bins + 1)]
    bin_totals = [0] * n_bins
    bin_pred_sum = [0.0] * n_bins
    bin_label_sum = [0.0] * n_bins

    for p, y in zip(probs, labels):
        b = min(int(p * n_bins), n_bins - 1)
        bin_totals[b] += 1
        bin_pred_sum[b] += p
        bin_label_sum[b] += y

    avg_pred = []
    avg_obs = []
    bin_centers = []

    for i in range(n_bins):
        center = (bins[i] + bins[i+1]) / 2
        bin_centers.append(center)

        if bin_totals[i] > 0:
            avg_pred.append(bin_pred_sum[i] / bin_totals[i])
            avg_obs.append(bin_label_sum[i] / bin_totals[i])
        else:
            avg_pred.append(None)
            avg_obs.append(None)

    return bin_centers, avg_pred, avg_obs, probs, labels


def plot_calibration_curve(bin_centers, avg_pred, avg_obs):
    # Filter out empty bins (where avg_pred or avg_obs is None)
    xs = []
    ys_pred = []
    ys_obs = []
    for c, p, o in zip(bin_centers, avg_pred, avg_obs):
        if p is not None and o is not None:
            xs.append(c)
            ys_pred.append(p)
            ys_obs.append(o)

    plt.figure()
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    # Model calibration points/line
    plt.plot(xs, ys_obs, marker="o", label="Observed frequency")
    plt.plot(xs, ys_pred, marker="x", linestyle=":", label="Avg predicted prob")

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed fraction of positives")
    plt.title("Calibration Curve (Reliability Plot)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def give_user_prediction(theta_list):
  with open('Mental-Illness-Test-Anxiety.csv', mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    header = list(header)
    questions = header[:-1]
  all_answers = []
  
  for question in questions:
    answer = input(f"{question} (0 = No, 1 = Yes): ").strip()
    while True:
      if answer in ("0", "1"):
        answer = int(answer)
        break
      else:
        print("Please enter 0 or 1.")
        answer = input(f"{question} (0 = No, 1 = Yes): ").strip()
    all_answers.append(answer)
  
  sigmoid = 1 / (1 + math.exp(-dot_prod(theta_list, all_answers)))
  print(sigmoid)
  
  # gives a prediction of depression disorder to the user
  if sigmoid >= 0.85:
    print("Very High Possibility of Anxiety Disorder")
  elif sigmoid >= 0.6:
    print("High Possibility of Anxiety Disorder")
  elif sigmoid >= 0.4:
    print("Moderate Possibility of Anxiety Disorder")
  else:
    print("Low Possibility of Anxiety Disorder")


def main():
  theta_list = logistic_regression_trainer()
  predictions_and_tests = logistic_regression_tester(theta_list)
  predictions = predictions_and_tests[0]
  tests = predictions_and_tests[1]
  accuracy(predictions, tests)
  
  bin_centers, avg_pred, avg_obs, probs, labels = calibration_curve(theta_list, tests)

  print("\n--- Calibration Curve Data ---")
  for c, p, o in zip(bin_centers, avg_pred, avg_obs):
      if p is not None:
          print(f"Bin {c:.2f}: Avg predicted = {p:.3f}, Avg observed = {o:.3f}")

  # Brier score
  brier = sum((p - y)**2 for p, y in zip(probs, labels)) / len(labels)
  print(f"\nBrier Score: {brier:.4f}")

  # Show the calibration plot
  plot_calibration_curve(bin_centers, avg_pred, avg_obs)
  
  #give_user_prediction(theta_list)


if __name__ == "__main__":
  main()