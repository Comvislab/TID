import numpy as np
import matplotlib.pyplot as plt

# Example confusion matrix for 3 classes
# Rows are actual classes, columns are predicted classes

def CinsRocRecPrec(conMat):

    # Number of classes
    num_classes = conMat.shape[0]

    # Initialize precision, recall, f1-score lists
    precision = []
    recall = []
    f1_score = []

    # Initialize TPR and FPR lists
    tpr = []
    fpr = []


    # Calculate metrics for each class
    for i in range(num_classes):
        tp = conMat[i, i]
        fn = conMat[i, :].sum() - tp
        fp = conMat[:, i].sum() - tp
        tn = conMat.sum() - (tp + fn + fp)
        
        precision_i = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score_i = 2 * (precision_i * recall_i) / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0
        
        precision.append(precision_i)
        recall.append(recall_i)
        f1_score.append(f1_score_i)

        tpr_i = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_i = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr.append(tpr_i)
        fpr.append(fpr_i)


    # Macro-average
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1_score)

    # Micro-average
    micro_precision = np.sum(conMat.diagonal()) / np.sum(conMat)
    micro_recall = micro_precision  # Same as micro precision in this context
    micro_f1 = micro_precision  # Same as micro precision in this context

    # Weighted-average
    support = conMat.sum(axis=1)  # Number of true instances per class
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1_score, weights=support)

    # Output results
    print(f"Precision (Macro): {macro_precision:.2f}")
    print(f"Recall (Macro): {macro_recall:.2f}")
    print(f"F1-Score (Macro): {macro_f1:.2f}")

    print(f"Precision (Micro): {micro_precision:.2f}")
    print(f"Recall (Micro): {micro_recall:.2f}")
    print(f"F1-Score (Micro): {micro_f1:.2f}")

    print(f"Precision (Weighted): {weighted_precision:.2f}")
    print(f"Recall (Weighted): {weighted_recall:.2f}")
    print(f"F1-Score (Weighted): {weighted_f1:.2f}")


    # Plot ROC curve --> (APPROXIMATE) <----
    fig=plt.figure()
    colors = ['blue', 'brown', 'magenta', 'red', 'green']
    for i in range(num_classes):
        plt.scatter(fpr[i], tpr[i], color=colors[i], label=f'Class {i}')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line (random chance)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Approximate ROC Points from Confusion Matrix')
    plt.legend(loc="lower right")
    #plt.show()
    return fig

