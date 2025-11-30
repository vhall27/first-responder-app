# train_and_save.py
import math
import csv
import json

TRAIN_FILE = "csv-files/Mental-Illness-Train-Depression.csv"

def dot_prod(theta_list, example):
    total = theta_list[0]  # bias term
    for i in range(1, len(theta_list)):
        total += theta_list[i] * float(example[i - 1])
    return total

def logistic_regression_trainer():
    with open(TRAIN_FILE, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        examples = list(csv_reader)

    # number of elements in theta/gradient without label column
    m_elems = len(header) - 1
    elems_with_bias = m_elems + 1

    # thetas, which include theta_0
    theta_list = [0.0] * elems_with_bias

    step_size = 0.0001
    iterations = 1000

    for _ in range(iterations):
        gradient_list = [0.0] * elems_with_bias

        for example in examples:
            y = int(example[-1])
            z = dot_prod(theta_list, example)
            sigmoid = 1 / (1 + math.exp(-z))
            second_term = y - sigmoid

            # bias gradient
            gradient_list[0] += second_term

            # other gradients
            for j in range(1, elems_with_bias):
                first_term = float(example[j - 1])
                gradient_list[j] += first_term * second_term

        # gradient ascent
        for j in range(len(theta_list)):
            theta_list[j] += step_size * gradient_list[j]

    return theta_list

def main():
    theta_list = logistic_regression_trainer()
    with open("theta_depression.json", "w") as f:
        json.dump(theta_list, f)
    print("Saved theta_list to theta_depression.json")

if __name__ == "__main__":
    main()
