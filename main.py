import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import spacy
import string
# nltk.download('punkt_tab')
# nltk.download('stopwords')

df1 = pd.read_csv('rotten_tomatoes_movies.csv')
df2 = pd.read_csv('train.csv')
df3 = pd.read_csv('rotten_tomatoes_movie_reviews.csv')

rotten_merged = pd.merge(df1, df3, on='id', how='inner')
rotten_merged = rotten_merged.drop_duplicates(subset='id', keep='first')
df2.rename(columns={'movie_name': 'title'}, inplace=True)
merged_df = pd.merge(rotten_merged, df2, on='title', how='inner')

no_duplicates_df = merged_df.drop_duplicates(subset='title', keep='first').copy()

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

if 'reviewText' in no_duplicates_df.columns:
    def preprocess_text(text):
        # na tokeny
        tokens = word_tokenize(text)
        # usuwanie interpunkcji
        tokens = [token for token in tokens if token not in string.punctuation]
        # stop worlds removal
        tokens = [token for token in tokens if token.lower() not in stop_words]
        # stemming
        tokens = [stemmer.stem(token) for token in tokens]
        return tokens

    no_duplicates_df['tokenized_review'] = no_duplicates_df['reviewText'].astype(str).apply(preprocess_text)
    print(no_duplicates_df[['title', 'reviewText', 'tokenized_review']].head())
    print(no_duplicates_df)
    no_duplicates_df.to_csv('new_database.csv', sep=',', encoding='utf-8')
else:
    print("Kolumna 'reviewText' nie istnieje w no_duplicates_df. Wybierz inną kolumnę do tokenizacji.")