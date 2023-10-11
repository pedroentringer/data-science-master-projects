import pandas as pd
import re
import nltk
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

#nltk.download('omw-1.4')
#nltk.download('wordnet')
#nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()
english_stopwords = list(stopwords.words('english'))

df = pd.read_csv('../datasets/twitter_training.csv', header=None)
df.columns = ['id','topic','sentiment','tweet']
df.dropna(inplace=True, axis=0)
df = df[0:10]


def remove_stopwords(text):
    for z in english_stopwords:
        text = re.sub(r'\b' + z + r'\b', "", text, flags=re.IGNORECASE)
    return text


def preprocessing(text):

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"(?i)\b((?:https?:\/|pic\.|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", "", text)

    text = remove_stopwords(text)

    text = text.strip().rstrip()

    return text


def text_lemmatise(text):
    doc = nlp(text)
    list_lemmas = []
    for token in doc:
        list_lemmas.append(token.lemma_)
    list_lemmas = list(filter(str.strip, list_lemmas))

    sentence = ""
    for z in list_lemmas:
        sentence = sentence + " " + str(z)
        sentence = sentence.strip()

    return sentence


def clean_text(x):
    x = preprocessing(x)
    x = text_lemmatise(x)
    return x


def get_unique_labels(x):
    return list(set(entity.label_ for entity in x.ents))


def fn_counts(row):
    x = row['tweet_nlp']
    row_labels = row['tweet_labels']

    pos_counts = {
        'nouns': sum([1 for token in x if token.pos_ == 'NOUN']),
        'verbs': sum([1 for token in x if token.pos_ == 'VERB']),
        'adjectives': sum([1 for token in x if token.pos_ == 'ADJ']),
        'positive': sum([1 for token in x if sia.polarity_scores(token.text)['pos'] > 0]),
        'negative': sum([1 for token in x if sia.polarity_scores(token.text)['neg'] > 0]),
    }

    label_counts = {}
    for label in row_labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    return {**pos_counts, **label_counts}


def generate_word_ngrams(X, word_max_features_in, word_min_df_in, word_max_df_in, wordgram_range_in, verbose, feature_objetive):
    count_vect_word = CountVectorizer(
        analyzer='word',
        lowercase=True,
        max_features=word_max_features_in,
        min_df=word_min_df_in,
        max_df=word_max_df_in,
        ngram_range=wordgram_range_in
    )
    result_w = count_vect_word.fit_transform(X[feature_objetive])

    if (verbose):
        print(str(result_w.shape))

    X_train_ngrams=count_vect_word.transform(X[feature_objetive])
    list_n_grams=count_vect_word.get_feature_names_out()

    return (X_train_ngrams,list_n_grams)


df['tweet_cleaned'] = df['tweet'].apply(lambda x: clean_text(x))
df['tweet_nlp'] = df['tweet_cleaned'].apply(lambda x: nlp(x))
df['tweet_labels'] = df['tweet_nlp'].apply(lambda x: get_unique_labels(x))
df['tweet_labels_count'] = df.apply(lambda x: fn_counts(x), axis=1)

X_train_ngrams, list_n_grams = generate_word_ngrams(X=df,
                                                    word_max_features_in=None,
                                                    word_min_df_in=1,
                                                    word_max_df_in=0.5,
                                                    wordgram_range_in=(1,5),
                                                    verbose=True,
                                                    feature_objetive='tweet_cleaned')

print(X_train_ngrams.shape)

print(list_n_grams)

df['tweet_labels_count'].apply(lambda x: print(x))
