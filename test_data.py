import numpy as np
import pandas as pd
import os
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score
from scipy.stats import gmean


def k_fold_evaluation(y: np.ndarray, embeddings: np.ndarray, skf, classifier):
    balanced_accuracies = []
    g_means = []
    f1_scores = []

    for i, (train_index, test_index) in enumerate(skf.split(embeddings, y)):
        print(f"\nRepeat {i}")

        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        bal_acc = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracies.append(bal_acc)
        # print(f"Balanced Accuracy: {bal_acc:.4f}")

        recalls = recall_score(y_test, y_pred, average=None)

        g_mean_value = gmean(recalls)
        g_means.append(g_mean_value)
        # print(f"G-Mean: {g_mean_value:.4f}")

        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
        # print(f"F1-score: {f1:.4f}")

    return balanced_accuracies, g_means, f1_scores


def print_results(balanced_accuracies_first, g_means_first, f1_scores_first, balanced_accuracies, g_means, f1_scores):
    print('\n \nWyniki dla klasyfikacji na recenzjach: ')
    print(f"Średnia Balanced Accuracy: {np.mean(balanced_accuracies_first):.4f}")
    print(f"Średnia G-Mean: {np.mean(g_means_first):.4f}")
    print(f"Średni F1-score: {np.mean(f1_scores_first):.4f}")

    print('\nWyniki dla klasyfikacji na recenzjach i opisach: ')
    print(f"Średnia Balanced Accuracy: {np.mean(balanced_accuracies):.4f}")
    print(f"Średnia G-Mean: {np.mean(g_means):.4f}")
    print(f"Średni F1-score: {np.mean(f1_scores):.4f}")

    return [np.mean(balanced_accuracies_first), np.mean(g_means_first), np.mean(f1_scores_first)], [np.mean(balanced_accuracies), np.mean(g_means), np.mean(f1_scores)]


def save_all_to_files(csv_path, genre_unmerged, genre_merged, sentiment_unmerged, sentiment_merged, clf):
    save_to_file(csv_path, 'grenre_unmerged', genre_unmerged, clf)
    save_to_file(csv_path, 'genre_merged', genre_merged, clf)
    save_to_file(csv_path, 'sentiment_unmerged', sentiment_unmerged, clf)
    save_to_file(csv_path, 'sentiment_merged', sentiment_merged, clf)


def save_to_file(csv_path, name, data, clf):
    resutl_row = {
        'Classifier': clf,
        'Balanced accuracy': round(data[0], 3),
        'G-Mean': round(data[1], 3),
        'F1-score': round(data[2], 3)
    }
    file_path = str(os.path.join(csv_path + name + '.csv'))
    pd.DataFrame([resutl_row]).to_csv(file_path, mode='a', header=not pd.io.common.file_exists(file_path), index=False)
