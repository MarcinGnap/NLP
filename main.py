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
import pickle
import os
from load_n_prep_data import load_n_merge_data, encode_text_unmerged, df_cleanup_for_unmerged, df_cleanup_for_merged, encode_text_merged
from test_data import print_results, k_fold_evaluation

# nltk.download('punkt_tab')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
skf = StratifiedKFold(n_splits=5, shuffle=True)

merged_df = load_n_merge_data()
no_duplicates_df = df_cleanup_for_unmerged(merged_df)
y, embeddings = encode_text_unmerged(no_duplicates_df)

# Stratyfikowana walidacja krzyżowa (tak idę z komentarzami od dołu to nie piszę wszystkiego chyba)

balanced_accuracies_first, g_means_first, f1_scores_first = k_fold_evaluation(y, embeddings, skf)

# z łączeniem cech

no_duplicates_df = df_cleanup_for_merged(merged_df)

y, embeddings = encode_text_merged(no_duplicates_df)

balanced_accuracies_second, g_means_second, f1_scores_second = k_fold_evaluation(y, embeddings, skf)

print_results(balanced_accuracies_first, g_means_first, f1_scores_first, balanced_accuracies_second, g_means_second, f1_scores_second)