import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Define the confusion matrix
confusion_matrix = np.array([
    [27,  0,  0,  0,  0],
    [ 0, 25,  0,  0,  0],
    [ 0,  0, 42,  0,  0],
    [ 0,  0,  0, 50,  0],
    [ 0,  2,  0,  4,  4]
])

# Derive true labels and predictions for metric calculations
true_labels = []
predicted_labels = []

for i in range(len(confusion_matrix)):
    for j in range(len(confusion_matrix[i])):
        true_labels.extend([i] * confusion_matrix[i, j])
        predicted_labels.extend([j] * confusion_matrix[i, j])

# Calculate metrics
report = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)

# Convert to readable format for individual class metrics
class_metrics = {f"Class {i+1}": {
    "Precision": report[str(i)]["precision"],
    "Recall": report[str(i)]["recall"],
    "F1-Score": report[str(i)]["f1-score"]}
    for i in range(len(confusion_matrix))
}

# Calculate macro and weighted averages
macro_avg = report["macro avg"]
weighted_avg = report["weighted avg"]

print(class_metrics)
print("Macro Average")
print("-------------")
print(macro_avg)
print("*************")

print("Weighted Average")
print("-------------")
print(weighted_avg)
