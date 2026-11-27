
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score


def dataset_to_features(dataset):
    all_features = []
    all_labels = []

    for images, labels in dataset:
        images = tf.image.rgb_to_grayscale(images)
        images = tf.image.resize(images, [16, 16])
        images = images / 255.0
        images = tf.reshape(images, [tf.shape(images)[0], 256])

        all_features.append(images.numpy())
        all_labels.append(tf.argmax(labels, axis=1).numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def run_logistic_baseline(X_train, y_train, X_val, y_val, class_weight=None):

    model = LogisticRegression(
        max_iter=300,
        random_state=42,
        class_weight=class_weight
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    print("Validation Accuracy:", accuracy_score(y_val, predictions))
    print("Validation Macro F1:", f1_score(y_val, predictions, average="macro"))
    print("Balanced Accuracy:", balanced_accuracy_score(y_val, predictions))

    print(classification_report(y_val, predictions))

    return model, predictions
