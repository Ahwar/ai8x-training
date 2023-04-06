import numpy as np
import pandas as pd

classes = (
    "up",
    "down",
    "left",
    "right",
    "stop",
    "go",
    "yes",
    "no",
    "on",
    "off",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "zero",
    "UNKNOWN",
)

# below is the confusion matrix
# rows are actual labels
# columns are detected labels
# the indices are classes
# you can map them to classes labels defined in classes variable
exp2 = "logs/evaluate_exp002_1unknown_kws20_v3_400e_adamo_0-001lr_256b_nobias_KWS_20dataset___2023.03.30-145730/confusion_matrices/confusion_matrix_epoch_-1.npy"
exp4 = "logs/evaluate_exp004_0unknown_kws20_v3_400e_adamo_0-001lr_256b_nobias_KWS_20dataset___2023.03.31-083133/confusion_matrices/confusion_matrix_epoch_-1.npy"
res = "logs/evaluate_resume_kws20_v3_400e_adamo_0-001lr_256b_nobias_KWS20dataset___2023.04.03-080130/confusion_matrices/confusion_matrix_epoch_-1.npy"
res260 = "logs/evaluateepoch250_resume_kws20_v3_400e_adamo_0-001lr_256b_nobias_KWS20dataset___2023.04.05-131247/confusion_matrices/confusion_matrix_epoch_-1.npy"
matrix_ = np.load(res260)


print("====" * 10)


def calculate_accuracies(matrix, unknown_is_ignored=False):
    """calculate overall accuracy of a confusion matrix

    Args:
        matrix (np.ndarray): the confusion matrix with actual classes on y-axis,
                            and predicted classes on x-axis
        unknown_is_ignored (bool, optional): if you want to ignore last index to
                            ignore unknown class. Defaults to False.

    Returns:
        accuracy (float): the overall accuracy of the confusion matrix
    """

    if unknown_is_ignored:
        matrix = np.delete(matrix, -1, 1)
        matrix = np.delete(matrix, -1, 0)
    true_positives = 0
    for cls, val in enumerate(matrix):
        true_positives += val[cls]
    return true_positives / np.cumsum(matrix)[-1]


def calculate_precision(matrix, unknown_is_ignored=False):
    """calculate per class precision of a confusion matrix
        precision is defined as out of all the points predicted to be positive how many were actually positive

    Args:
        matrix (np.ndarray): the confusion matrix with actual classes on y-axis,
                            and predicted classes on x-axis
        unknown_is_ignored (bool, optional): if you want to ignore last index to
                            ignore unknown class. Defaults to False.

    Returns:
        precisions (list): the list of precisions by each class
    """
    if unknown_is_ignored:
        matrix = np.delete(matrix, -1, 1)
        matrix = np.delete(matrix, -1, 0)
    tp_fp_sum = np.sum(matrix, axis=0)
    precisions = []
    for cls, val in enumerate(matrix):
        true_positive = matrix[cls][cls]
        precisions.append((true_positive / tp_fp_sum[cls]))

    return precisions


def calculate_recall(matrix, unknown_is_ignored=False):
    """calculate per class recall of a confusion matrix
        recall is defined as out positives, how many did we correctly classifed as positives

    Args:
        matrix (np.ndarray): the confusion matrix with actual classes on y-axis,
                            and predicted classes on x-axis
        unknown_is_ignored (bool, optional): if you want to ignore last index to
                            ignore unknown class. Defaults to False.

    Returns:
        recall (list): the list of recall by each class
    """

    if unknown_is_ignored:
        matrix = np.delete(matrix, -1, 1)
        matrix = np.delete(matrix, -1, 0)
    recalls = []
    for cls, val in enumerate(matrix):
        recall = val[cls] / sum(val)
        recalls.append(recall)
    return recalls


print("====" * 8, "\nAccuracy")

accuracies_with_unknown = calculate_accuracies(matrix_)
accuracies_without_unknown = calculate_accuracies(matrix_, unknown_is_ignored=True)

print("overall accuracy with unknown", accuracies_with_unknown)
print(
    "overall accuracy without unknown",
    accuracies_without_unknown,
)
print("====" * 8, "\nPrecision")
precisions_with_unknown = calculate_precision(matrix_)
precisions_without_unknown = calculate_precision(matrix_, unknown_is_ignored=True)

print("Per class precisions with unknown", precisions_with_unknown)
print(
    "overall precisions with unknown", sum(precisions_with_unknown) / len(precisions_with_unknown)
)

print("\n\n\nPer class precisions without unknown", precisions_without_unknown)
print(
    "overall precisions without unknown",
    sum(precisions_without_unknown) / len(precisions_without_unknown),
)


print("====" * 8, "\nRecall")

recall_with_unknown = calculate_recall(matrix_)
recall_without_unknown = calculate_recall(matrix_, unknown_is_ignored=True)

print("Per class recall with unknown", recall_with_unknown)
print("overall recall with unknown", sum(recall_with_unknown) / len(recall_with_unknown))


print("\n\n\nPer class recall without unknown", recall_without_unknown)
print(
    "overall recall without unknown",
    sum(recall_without_unknown) / len(recall_without_unknown),
)


def calculate_f1_score(beta, precisions, recalls):
    f1s = []
    for i in range(len(precisions)):
        p = precisions[i]
        r = recalls[i]
        f1 = (1 + beta) * ((p * r) / ((beta * p) + r))
        f1s.append(f1)
    return f1s


beta = 0.5
dictionary = {
    "classes ": classes,
    "precisions_with_unknown": precisions_with_unknown,
    "recall_with_unknown": recall_with_unknown,
    "f1 score with unknown": calculate_f1_score(
        beta, precisions_with_unknown, recall_with_unknown
    ),
    "precisions_without_unknown": precisions_without_unknown + [np.nan],
    "recall_without_unknown": recall_without_unknown + [np.nan],
    "f1 score without unknown": calculate_f1_score(
        beta, precisions_without_unknown, recall_without_unknown
    )
    + [np.nan],
}

df = pd.DataFrame(dictionary)
# calculate the mean values of each column
mean_values = df.mean(axis=0)

# append the mean values as a new row to the DataFrame
df = df.append(mean_values, ignore_index=True)
df = df.round(5)
out = "resume260_model_evaluation_stats"
df.to_excel(out + ".xlsx", index=False)
df.to_csv(out + ".csv", index=False)
df.to_html(out + ".html", index=False)
