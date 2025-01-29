import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import numpy as np

def load_n_merge_data() -> DataFrame:
    df1 = pd.read_csv('csv/rotten_tomatoes_movies.csv')
    df2 = pd.read_csv('csv/train.csv')
    df3 = pd.read_csv('csv/rotten_tomatoes_movie_reviews.csv')

    rotten_merged = pd.merge(df1, df3, on='id', how='inner')
    rotten_merged = rotten_merged.drop_duplicates(subset='id', keep='first')
    rotten_merged_selected = rotten_merged[['title', 'reviewText', 'scoreSentiment']]
    rotten_merged_selected.rename(columns={'scoreSentiment': 'sentiment'}, inplace=True)
    df2.rename(columns={'movie_name': 'title'}, inplace=True)
    df2 = df2[['title', 'genre', 'synopsis']]
    merged_df = pd.merge(rotten_merged_selected, df2, on='title', how='inner')

    return merged_df


def df_cleanup_for_unmerged(merged_df: DataFrame) -> DataFrame:
    no_duplicates_df = merged_df.drop_duplicates(subset='title', keep='first').copy()
    no_duplicates_df = no_duplicates_df.dropna(subset=['reviewText'])
    no_duplicates_df['reviewText'] = no_duplicates_df['reviewText'].astype(str)

    return no_duplicates_df


def df_cleanup_for_merged(merged_df: DataFrame) -> DataFrame:
    no_duplicates_df = merged_df.drop_duplicates(subset='title', keep='first').copy()

    no_duplicates_df = no_duplicates_df.dropna(subset=['reviewText', 'synopsis'])
    no_duplicates_df['reviewText'] = no_duplicates_df['reviewText'].astype(str)
    no_duplicates_df['synopsis'] = no_duplicates_df['synopsis'].astype(str)

    return no_duplicates_df


def encode_text_unmerged(no_duplicates_df: DataFrame, y_column: str = 'genre'):
    # y_column - 'genre' or 'sentiment'
    df_sentences = no_duplicates_df['reviewText'].dropna().astype(str).tolist()
    labels = no_duplicates_df[y_column].dropna().tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df_sentences, convert_to_numpy=True)

    return y, embeddings


def encode_text_merged(no_duplicates_df: DataFrame, y_column: str = 'genre'):
    # y_column - 'genre' or 'sentiment'
    df_reviews = no_duplicates_df['reviewText'].tolist()
    df_synopsis = no_duplicates_df['synopsis'].tolist()
    labels = no_duplicates_df[y_column].tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    review_embeddings = model.encode(df_reviews, convert_to_numpy=True)
    synopsis_embeddings = model.encode(df_synopsis, convert_to_numpy=True)
    embeddings = np.hstack((review_embeddings, synopsis_embeddings))

    return y, embeddings


def save_to_npy(arr: np.ndarray, name: str):
    np.save(name, arr)


def load_from_npy(name: str) -> np.ndarray:
    arr = np.load(f'{name}.npy')
    return arr

def init_load(no_duplicates_df, merged_df):
    y, embeddings = encode_text_unmerged(no_duplicates_df, 'genre') #genre, sentiment
    save_to_npy(embeddings, 'npy_files/genre_embeddings_unmerged')
    save_to_npy(y, 'npy_files/genre_labels_unmerged')
    y_s, embeddings_s = encode_text_unmerged(no_duplicates_df, 'sentiment')
    save_to_npy(embeddings_s, 'npy_files/sentiment_embeddings_unmerged')
    save_to_npy(y_s, 'npy_files/sentiment_labels_unmerged')
    y_m, embeddings_s = encode_text_merged(no_duplicates_df, 'genre')
    save_to_npy(embeddings, 'npy_files/genre_embeddings_merged')
    save_to_npy(y, 'npy_files/genre_labels_merged')
    y_sm, embeddings_sm = encode_text_merged(no_duplicates_df, 'sentiment')
    save_to_npy(embeddings_sm, 'npy_files/sentiment_embeddings_merged')
    save_to_npy(y_sm, 'npy_files/sentiment_labels_merged')
    return df_cleanup_for_merged(merged_df)
