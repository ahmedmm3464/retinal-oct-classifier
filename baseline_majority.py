
import pandas as pd
from sklearn.metrics import accuracy_score

def run_majority_baseline(train_labels, val_labels):
    majority_class = train_labels.mode()[0]

    predictions = [majority_class] * len(val_labels)

    accuracy = accuracy_score(val_labels, predictions)

    print("Majority Class:", majority_class)
    print("Validation Accuracy:", accuracy)

    return predictions
