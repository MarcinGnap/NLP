######################################################################################################
# bez łączenia cech
######################################################################################################

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score
from scipy.stats import gmean
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# nltk.download('punkt_tab')
# nltk.download('stopwords')

df1 = pd.read_csv('rotten_tomatoes_movies.csv')
df2 = pd.read_csv('train.csv')
df3 = pd.read_csv('rotten_tomatoes_movie_reviews.csv')

rotten_merged = pd.merge(df1, df3, on='id', how='inner')
rotten_merged = rotten_merged.drop_duplicates(subset='id', keep='first')
rotten_merged_selected = rotten_merged[['title', 'reviewText', 'scoreSentiment']]
df2.rename(columns={'movie_name': 'title'}, inplace=True)
df2 = df2[['title', 'genre', 'synopsis']]
merged_df = pd.merge(rotten_merged_selected, df2, on='title', how='inner')

no_duplicates_df = merged_df.drop_duplicates(subset='title', keep='first').copy()

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

no_duplicates_df = no_duplicates_df.dropna(subset=['reviewText'])

no_duplicates_df['reviewText'] = no_duplicates_df['reviewText'].astype(str)

df_bert = no_duplicates_df[['reviewText', 'genre']]


# przygotowanie danych
df_sentences = no_duplicates_df['reviewText'].dropna().astype(str).tolist()
labels = no_duplicates_df['genre'].dropna().tolist()

# kodowanie naszych gatunków czyli labelów
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# tekst cech na embeddingi
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df_sentences, convert_to_numpy=True)

# Stratyfikowana walidacja krzyżowa (tak idę z komentarzami od dołu to nie piszę wszystkiego chyba)
skf = StratifiedKFold(n_splits=5, shuffle=True)

balanced_accuracies_first = []
g_means_first = []
f1_scores_first = []

for i, (train_index, test_index) in enumerate(skf.split(embeddings, y)):
    print(f"\nFold {i}")

    # podział na foldy
    X_train, X_test = embeddings[train_index], embeddings[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # trening
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # predykcja na teście
    y_pred = classifier.predict(X_test)

    # balanced accuracy
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    balanced_accuracies_first.append(bal_acc)
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    # czułość dla każdej klasy
    recalls = recall_score(y_test, y_pred, average=None)

    # średnia geometryczna czułości
    g_mean_value = gmean(recalls)
    g_means_first.append(g_mean_value)
    print(f"G-Mean: {g_mean_value:.4f}")

    # F1-score
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores_first.append(f1)
    print(f"F-1 score: {f1:.4f}")


######################################################################################################
# z łączeniem cech
######################################################################################################

no_duplicates_df = merged_df.drop_duplicates(subset='title', keep='first').copy()

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# usuwanie wybrakowanych rekordów
no_duplicates_df = no_duplicates_df.dropna(subset=['reviewText', 'synopsis'])
no_duplicates_df['reviewText'] = no_duplicates_df['reviewText'].astype(str)
no_duplicates_df['synopsis'] = no_duplicates_df['synopsis'].astype(str)

# pierwsze przygotowanie cech
df_reviews = no_duplicates_df['reviewText'].tolist()
df_synopsis = no_duplicates_df['synopsis'].tolist()
labels = no_duplicates_df['genre'].tolist()

# kodowanie etykiet, czyli u nas gatunków
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# nasz Bercik
model = SentenceTransformer("all-MiniLM-L6-v2")

# no tu tworzymy te embeddingi, co pisałem o nich z cech
review_embeddings = model.encode(df_reviews, convert_to_numpy=True)
synopsis_embeddings = model.encode(df_synopsis, convert_to_numpy=True)

# łączenie cech (niby mówi się na to konkatenacja embeddingów)
embeddings = np.hstack((review_embeddings, synopsis_embeddings))

balanced_accuracies = []
g_means = []
f1_scores = []

for i, (train_index, test_index) in enumerate(skf.split(embeddings, y)):
    print(f"\nFold {i}")

    # podział danych na foldy
    X_train, X_test = embeddings[train_index], embeddings[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # trening klasyfikatora
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # predykcja na zbiorze testowym
    y_pred = classifier.predict(X_test)

    # balanced Accuracy
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    balanced_accuracies.append(bal_acc)
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    # czułość dla każdej klasy
    recalls = recall_score(y_test, y_pred, average=None)

    # średnia geometryczna czułości (mogłem źle policzyć, ale naprawię później)
    g_mean_value = gmean(recalls)
    g_means.append(g_mean_value)
    print(f"G-Mean: {g_mean_value:.4f}")

    # F1-score - póśniej przerobimy na jakieś ef beta skor
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores.append(f1)
    print(f"F1-score: {f1:.4f}")




print('\n \nWyniki dla klasyfikacji na recenzjach: ')
print(f"Średnia Balanced Accuracy: {np.mean(balanced_accuracies_first):.4f}")
print(f"Średnia G-Mean: {np.mean(g_means_first):.4f}")
print(f"Średni F1-score: {np.mean(f1_scores_first):.4f}")

print('\nWyniki dla klasyfikacji na recenzjach i opisach: ')
print(f"Średnia Balanced Accuracy: {np.mean(balanced_accuracies):.4f}")
print(f"Średnia G-Mean: {np.mean(g_means):.4f}")
print(f"Średni F1-score: {np.mean(f1_scores):.4f}")