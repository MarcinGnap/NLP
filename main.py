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
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
# from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings
from load_n_prep_data import load_n_merge_data, encode_text_unmerged, df_cleanup_for_unmerged, df_cleanup_for_merged, \
    encode_text_merged, save_to_npy, load_from_npy, init_load
from test_data import print_results, k_fold_evaluation, save_all_to_files
import warnings
warnings.filterwarnings('ignore')
import nltk

INIT_RUN = False


if INIT_RUN:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
# one-vs-all ze stratified k-fold
# ładowanie do .parquet
# można ze stemmingiem i bez
# w analizie senytmentu można dodać gatunek/tytuł/opis
# jak dobrze pójdzie to można na konferencję
# można potestować różne modele

merged_df = load_n_merge_data()
no_duplicates_df = df_cleanup_for_unmerged(merged_df)

if INIT_RUN:
    no_duplicates_df_merged = init_load(no_duplicates_df, merged_df)

y = load_from_npy('npy_files/genre_labels_unmerged')
y_s = load_from_npy('npy_files/sentiment_labels_unmerged')
y_m = load_from_npy('npy_files/genre_labels_merged')
y_sm = load_from_npy('npy_files/sentiment_labels_merged')

embeddings = load_from_npy('npy_files/genre_embeddings_unmerged')
embeddings_s = load_from_npy('npy_files/sentiment_embeddings_unmerged')
embeddings_m = load_from_npy('npy_files/genre_embeddings_merged')
embeddings_sm = load_from_npy('npy_files/sentiment_embeddings_merged')

clfs = {"gnb": GaussianNB(), "knn": KNeighborsClassifier(), "dtc": DecisionTreeClassifier(), "log": LogisticRegression(), "svc": SVC()}

for name, clf in clfs.items():
    genre_unmerged, genre_merged = print_results(*k_fold_evaluation(y, embeddings, skf, clf), *k_fold_evaluation(y_m, embeddings_m, skf, clf))
    sentiment_unmerged, sentiment_merged = print_results(*k_fold_evaluation(y_s, embeddings_s, skf, clf), *k_fold_evaluation(y_sm, embeddings_sm, skf, clf))
    save_all_to_files('./results/', genre_unmerged, genre_merged, sentiment_unmerged, sentiment_merged, name)
