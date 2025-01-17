import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from scipy.stats import gmean


def k_fold_evaluation(y: np.ndarray, embeddings: np.ndarray, skf, classifier):
    balanced_accuracies = []
    g_means = []
    f1_scores = []

    for i, (train_index, test_index) in enumerate(skf.split(embeddings, y)):
        # print(f"\nFold {i}")

        # podział danych na foldy
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # trening klasyfikatora
        # classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        # predykcja na zbiorze testowym
        y_pred = classifier.predict(X_test)

        # balanced Accuracy
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracies.append(bal_acc)
        # print(f"Balanced Accuracy: {bal_acc:.4f}")

        # czułość dla każdej klasy
        recalls = recall_score(y_test, y_pred, average=None)

        # średnia geometryczna czułości (mogłem źle policzyć, ale naprawię później)
        g_mean_value = gmean(recalls)
        g_means.append(g_mean_value)
        # print(f"G-Mean: {g_mean_value:.4f}")

        # F1-score - póśniej przerobimy na jakieś ef beta skor
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
