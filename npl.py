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
matrix_ = np.asarray(
   [[  960,    1,    8,    3,    7,     4,     0,    0,    6,   59,     1,     2,     1,     3,    5,     1,     5,     4,     1,    0,    9],
    [    4, 977,   0,   2,   7,   31,    1,   9,   7,   3,    4,    6,    0,    3,   5,    0,    7,    4,     5,   1,   13],
    [   12,   5, 953,  15,   3,    2,   31,   5,   0,   0,    1,    3,    7,    0,   6,    4,    4,    3,     5,   1,    2],
    [    1,   2,  36, 977,   1,    4,    2,   0,   0,   0,    8,    0,    2,    3,  30,    0,    0,    1,     8,   1,    7],
    [   22,   3,   0,   0,1018,   13,    3,   1,   1,   3,    1,    5,    0,    2,   6,    4,    6,    4,     3,   1,    2],
    [    5,  25,   1,   0,   4, 1025,    2,  14,   1,   3,    1,   13,    2,    9,   5,    1,    2,    0,     0,   1,    8],
    [    5,   3,   6,   0,   0,    5, 1049,   0,   1,   0,    2,    7,    4,    0,   2,    5,    1,    2,     0,   3,    3],
    [    7,  19,  15,   4,   2,   58,   14, 995,   1,   1,    3,    4,    2,    0,   1,    0,    0,    1,    30,  17,    8],
    [   14,   4,   2,   1,   0,    8,    0,   0, 960,  45,   16,    3,    5,   28,  25,    0,    6,    6,     4,   0,    4],
    [   74,   0,   5,   0,   4,    6,    7,   0,  24, 886,    0,    0,    1,   16,  10,    0,    1,    3,     0,   0,    4],
    [    1,   3,   8,   4,   0,    4,    7,   4,   8,   1,  988,    2,    1,   12,  10,    0,    4,    0,    10,   5,    2],
    [    2,   4,   1,   0,   0,   11,    5,   0,   0,   0,    1, 1041,   29,    9,   0,    2,    5,   16,     3,  15,    2],
    [    5,   4,   1,  14,   0,    6,    2,   2,   0,   0,    0,   40,  966,    3,   0,    5,    1,   14,     2,  11,   16],
    [    4,   0,   2,   1,   3,    8,    0,   2,   4,  10,    7,   21,    5,  972,   7,    5,    5,    5,     0,   3,   10],
    [    8,   1,   0,  21,   2,    0,    0,   0,  24,   7,    7,    0,    0,    3,1005,    0,    2,    2,    15,   1,   12],
    [    3,   0,   3,   1,   1,    2,   15,   0,   0,   0,    1,    6,    7,    5,   1, 1009,   15,   10,     1,   7,    5],
    [    3,   6,   1,   3,   7,    4,    3,   0,   1,   1,    1,    2,    6,    3,   3,    8, 1077,    1,     2,   2,   18],
    [    0,   1,   1,   6,   0,    5,    3,   0,   0,   0,    1,    7,   13,    6,   6,    7,    0, 1005,     0,   2,   11],
    [    0,   9,   3,  20,   0,    8,    1,  23,   2,   0,    9,    1,    0,    1,  15,    0,    5,    1,  1013,   1,    7],
    [    2,   1,   9,   1,   2,   12,   12,   6,   1,   0,    4,   21,    1,    6,   1,    4,   12,    2,     3,1066,    7],
    [  174, 197, 271, 135, 109,  238,  130, 127,  67,  99,  172,  163,  302,  382, 219,   72,  176,  101,   163, 245,11047]])


print("====" * 10)
print(np.sum(matrix_, axis=0))
def calculate_accuracies(matrix, unknown_is_ignored=False):
    """calculate per class accuracy of a confusion matrix 

    Args:
        matrix (np.ndarray): the confusion matrix with actual classes on y-axis,
                            and predicted classes on x-axis
        unknown_is_ignored (bool, optional): if you want to ignore last index to
                            ignore unknown class. Defaults to False.

    Returns:
        accuracy (list): the list of accuracy by each class
    """

    if unknown_is_ignored:
        matrix = np.delete(matrix, -1, 1)
        matrix = np.delete(matrix, -1, 0)
    accuracies = []
    for cls, val in enumerate(matrix):
        accuracy = (val[cls] / sum(val))
        accuracies.append(accuracy)
    return accuracies


def calculate_precision(matrix, unknown_is_ignored=False):
    """calculate per class precision of a confusion matrix 

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
        recall = (val[cls] / sum(val))
        recalls.append(recall)
    return recalls

print("====" * 8, "\nAccuracy")

accuracies_with_unknown = calculate_accuracies(matrix_)
accuracies_without_unknown = calculate_accuracies(matrix_, unknown_is_ignored=True)

print("All accuracies with unknown", accuracies_with_unknown)
print("overall accuracy with unknown", sum(accuracies_with_unknown) / len(accuracies_with_unknown))


print("\n\n\nAll accuracies without unknown", accuracies_without_unknown)
print(
    "overall accuracy without unknown",
    sum(accuracies_without_unknown) / len(accuracies_without_unknown),
)
print("====" * 8, "\nPrecision")
precisions_with_unknown = calculate_precision(matrix_)
precisions_without_unknown = calculate_precision(matrix_, unknown_is_ignored=True)

print("All precisions with unknown", precisions_with_unknown)
print("overall precisions with unknown", sum(precisions_with_unknown) / len(precisions_with_unknown))

print("\n\n\nAll precisions without unknown", precisions_without_unknown)
print(
    "overall precisions without unknown",
    sum(precisions_without_unknown) / len(precisions_without_unknown))


print("====" * 8, "\nRecall")

recall_with_unknown = calculate_recall(matrix_)
recall_without_unknown = calculate_recall(matrix_, unknown_is_ignored=True)

print("All recall with unknown", recall_with_unknown)
print("overall accuracy with unknown", sum(recall_with_unknown) / len(recall_with_unknown))


print("\n\n\nAll recall without unknown", recall_without_unknown)
print(
    "overall accuracy without unknown",
    sum(recall_without_unknown) / len(recall_without_unknown),
)

def calculate_f1_score(beta, precisions, recalls):
    f1s = []
    for i in range(len(precisions)):
        p = precisions[i]
        r = recalls[i]
        f1 = ((1 + beta) * ((p * r) / ((beta * p) + r)))
        f1s.append(f1)
    return f1s;

beta = 0.5
dictionary = {
    "classes " : classes,
    "accuracies_with_unknown" : accuracies_with_unknown,
    "precisions_with_unknown" : precisions_with_unknown,
    "recall_with_unknown" : recall_with_unknown,
    "f1 score with unknown": calculate_f1_score(beta, precisions_with_unknown, recall_with_unknown),
    "accuracies_without_unknown" : accuracies_without_unknown + [np.nan],
    "precisions_without_unknown" : precisions_without_unknown + [np.nan],
    "recall_without_unknown" : recall_without_unknown + [np.nan],
    "f1 score without unknown": calculate_f1_score(beta, precisions_without_unknown, recall_without_unknown) + [np.nan],
    }

df = pd.DataFrame(dictionary)
df.to_excel("model_evaluation_stats.xlsx")