import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import re
import time
from statistics import mean
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

from sentiment_analysis_spanish import sentiment_analysis
sentiment = sentiment_analysis.SentimentAnalysisSpanish()
import spacy as spacy
nlp = spacy.load('es_core_news_sm',  disable=["ner", "senter"])

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def compute_correlation_heat_map(dataset,target_feature):
    #method="pearson","kendall","spearman"
    correlations_pearson = dataset.corr(method="pearson")[[target_feature]].round(decimals=2)
    print(correlations_pearson)

    fig, ax = plt.subplots(figsize=(1, 44))
    sns.heatmap(correlations_pearson, annot=True, fmt='.2f',cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
    plt.savefig('heatmap_pearson.png', bbox_inches='tight', pad_inches=0.0)

def remove_stopwords(text):
    for z in list(stopwords.words('spanish')):
        text = re.sub(r'\b' + z + r'\b', "", text, flags=re.IGNORECASE)
    return text

def preprocessing(text):

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"(?i)\b((?:https?:\/|pic\.|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", "", text)

    text = remove_stopwords(text)

    text = text.strip().rstrip()

    return text

def lemmatise(text):
    doc = nlp(text)
    list_lemmas = []
    for token in doc:
        #print(token.lemma_ + " " + token.pos_)
        list_lemmas.append(token.lemma_)
    list_lemmas = list(filter(str.strip, list_lemmas))

    sentence = ""
    for z in list_lemmas:
        sentence = sentence + " " + str(z)
        sentence = sentence.strip()

    return sentence

def compute_features(x):

    x['Es_pol'] = sentiment.sentiment(x['Tweet'])

    x['Tweet_procesado'] = preprocessing(x['Tweet'])
    x['Tweet_procesado'] = lemmatise(x['Tweet_procesado']).lower()

    return x

def evaluate_CountVectorizer(X, y, analyzer):
    my_pipeline = Pipeline([
    ('ngrams', CountVectorizer(analyzer=analyzer)),
    ('clf', RandomForestClassifier(random_state=0))
    ])

    parameters = {}
    parameters.update({"CountVectorizer": {
        "ngrams__max_features": [50, 150, None],  # Primera ejecución sin este parámetro
        # luego fijar: 20%, 50%, None
        "ngrams__min_df": [0.001, 0.01, 0.1],
        "ngrams__max_df": [0.9, 0.7, 0.5],
        "ngrams__ngram_range": [(1, 2), (1, 4), (2, 4), (2, 5)]
    }})

    grid_search = GridSearchCV(my_pipeline, param_grid=parameters["CountVectorizer"], cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X, y)

    print(grid_search.best_params_)

def generate_char_ngrams(X, char_max_features_in, char_min_df_in, char_max_df_in, chargram_range_in,
                    verbose,feature_objetive):
    count_vect_char = CountVectorizer(
        analyzer='char',
        lowercase=True,
        max_features=char_max_features_in,
        min_df=char_min_df_in,
        max_df=char_max_df_in,
        ngram_range=chargram_range_in
    )
    result_c = count_vect_char.fit_transform(X[feature_objetive])

    if (verbose):
        print(str(result_c.shape))

    X_train_ngrams=count_vect_char.transform(X[feature_objetive])
    list_n_grams=count_vect_char.get_feature_names()

    return (X_train_ngrams,list_n_grams)

def generate_word_ngrams(X, word_max_features_in, word_min_df_in, word_max_df_in, wordgram_range_in,
                    verbose,feature_objetive):
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
    list_n_grams=count_vect_word.get_feature_names()

    return (X_train_ngrams,list_n_grams)

def generate_char_word_ngrams(X, char_max_features_in, char_min_df_in, char_max_df_in, chargram_range_in,
                    word_max_features_in, word_min_df_in, word_max_df_in, wordgram_range_in,
                    verbose,feature_objetive):
    count_vect_word = CountVectorizer(
        analyzer='word',
        lowercase=True,
        max_features=word_max_features_in,
        min_df=word_min_df_in,
        max_df=word_max_df_in,
        ngram_range=wordgram_range_in
    )
    result_w = count_vect_word.fit_transform(X[feature_objetive])

    count_vect_char = CountVectorizer(
        analyzer='char',
        lowercase=True,
        max_features=char_max_features_in,
        min_df=char_min_df_in,
        max_df=char_max_df_in,
        ngram_range=chargram_range_in
    )
    result_c = count_vect_char.fit_transform(X[feature_objetive])

    if (verbose):
        print(str(result_w.shape)
              + " " + str(result_c.shape))

    feature = FeatureUnion([("W", count_vect_word), ("C", count_vect_char)])
    X_train_ngrams=feature.transform(X[feature_objetive])
    list_n_grams=count_vect_word.get_feature_names()+count_vect_char.get_feature_names()

    return (X_train_ngrams,list_n_grams)

def add_params(X_train_ngrams, dataset, list_params_eval, list_n_grams):
    X_train_ngrams_aux = pd.DataFrame()

    for param in list_params_eval:
        X_train_ngrams_aux[param] = dataset[param]

    X_train_ngrams = pd.DataFrame(X_train_ngrams.toarray(),columns=list_n_grams)
    X_train_ngrams = X_train_ngrams.join(X_train_ngrams_aux)

    list_n_grams=[*list_n_grams,*list_params_eval]
    return (X_train_ngrams,list_n_grams)

def select_best_features(X_train,y_train, list_ngrams):
    sel = SelectFromModel(RandomForestClassifier(n_jobs=-1,random_state=0)) #n_jobs para paralelizarlo
    sel.fit(X_train, y_train)

    support = np.asarray(sel.get_support())
    columns_with_support = list(np.array(list_ngrams)[np.array(support)])

    X_train_best_features = X_train[columns_with_support]

    return X_train_best_features

def cross_validation(X, y):
    cls_nb = GaussianNB()
    cls_dt = DecisionTreeClassifier(random_state=0)
    cls_rf = RandomForestClassifier(random_state=0, n_jobs=-1)
    cls_rf2 = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight='balanced_subsample', criterion='gini', max_depth=200, max_features='log2', min_samples_leaf=1, min_samples_split=2, n_estimators=250)
    cls_bc = GradientBoostingClassifier(random_state=0)

    #clss = [cls_nb,cls_dt,cls_rf,cls_bc]
    clss = [cls_rf2]
    clss_name = ['NB', 'DT', 'RF', 'BC']
    print('starts ML')
    labels = [-1, 0, 1]
    for cls_index in range(len(clss)):
        time_init = time.time()
        print(clss[cls_index].__str__())
        y_pred = cross_val_predict(clss[cls_index], X, y, cv=10)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average=None, labels=labels)
        recall = recall_score(y, y_pred, average=None, labels=labels)
        f1 = f1_score(y, y_pred, average=None, labels=labels)

        accuracy = str("%.2f" % round(accuracy, 2))

        p_macro = str("%.2f" % round(mean(precision), 2))
        p_micro_neg = str("%.2f" % round(precision[0], 2))
        p_micro_neu = str("%.2f" % round(precision[1], 2))
        p_micro_pos = str("%.2f" % round(precision[2], 2))

        r_macro = str("%.2f" % round(mean(recall), 2))
        r_micro_neg = str("%.2f" % round(recall[0], 2))
        r_micro_neu = str("%.2f" % round(recall[1], 2))
        r_micro_pos = str("%.2f" % round(recall[2], 2))

        f_macro = str("%.2f" % round(mean(f1), 2))
        f_micro_neg = str("%.2f" % round(f1[0], 2))
        f_micro_neu = str("%.2f" % round(f1[1], 2))
        f_micro_pos = str("%.2f" % round(f1[2], 2))

        time_total = str("%.2f" % round(time.time()-time_init, 2))

        results = str("Accuracy: " + accuracy + "\n" +
        # "Precision macro: " + p_macro + "\n" +
        # "Precision neg: " + p_micro_neg + "\n" +
        # "Precision neu: " + p_micro_neu + "\n" +
        # "Precicion pos: " + p_micro_pos + "\n" +
        # "Recall macro: " + r_macro + "\n" +
        # "Recall neg: " + r_micro_neg + "\n" +
        # "Recall neu: " + r_micro_neu + "\n" +
        # "Recall pos: " + r_micro_pos + "\n" +
        # "F1 macro: " + f_macro + "\n" +
        # "F1 neg: " + f_micro_neg + "\n" +
        # "F1 neu: " + f_micro_neu + "\n" +
        # "F1 pos: " + f_micro_pos + "\n" +
        "Time: " + time_total)

        print(results)

def hyperparameter_optimisation(X_train, y_train):
    parameters = {}
    parameters.update({"Random Forest": {
        "classifier__n_estimators": [100, 250, 500], # [100, 150]
        "classifier__class_weight": ["balanced", "balanced_subsample", None], # ["balanced", None]
        "classifier__max_features": ["sqrt", "log2", None], # ["sqrt"]
        "classifier__max_depth": [10, 200, 1000, None], # [100, None]
        "classifier__min_samples_split": [2, 10, 100, None], # nao tem
        "classifier__min_samples_leaf": [1, 3, 30, None], # nao tem
        "classifier__criterion": ["gini", "entropy", "log_loss", None], # ["gini"]
        "classifier__random_state": [0] # só existe nos classificadores randomizados, então ele usa o mesmo valor sempre. Mesmo em produção podemos manter
    }})

    steps = [("classifier", RandomForestClassifier())]
    pipeline = Pipeline(steps=steps)
    param_grid = parameters["Random Forest"]

    # queremos ver primeiro o scoring como accuracy
    # entretando podemos mudar isso para a precision e recall
    gscv = GridSearchCV(pipeline, param_grid, n_jobs=-1, scoring="accuracy", verbose=True, cv=5)


    gscv.fit(X_train, y_train)
    # Get best parameters and score
    best_params = gscv.best_params_
    print(best_params)