# importando as bibliotecas necessárias
import time
from statistics import mean
import pandas as pd
import numpy as np
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

# Função que gera n-grams de palavras com base em um conjunto de dados de entrada e parâmetros especificados
# Parâmetros:
# - X: conjunto de dados de entrada
# - word_max_features_in: número máximo de features (palavras) a serem consideradas
# - word_min_df_in: frequência mínima com que uma palavra deve aparecer para ser considerada
# - word_max_df_in: frequência máxima com que uma palavra pode aparecer para ser considerada
# - wordgram_range_in: range de n-grams de palavras a serem gerados (exemplo: (1,2) para unigrams e bigrams)
# - verbose: se verdadeiro, exibe informações sobre o resultado da geração dos n-grams
# - feature_objetive: a feature do conjunto de dados a ser usada para gerar os n-grams


def generate_word_ngrams(X, word_max_features_in, word_min_df_in, word_max_df_in, wordgram_range_in, verbose,
                         feature_objetive):
    # Cria um objeto CountVectorizer com os parâmetros especificados
    count_vect_word = CountVectorizer(
        analyzer='word',
        lowercase=True,
        max_features=word_max_features_in,
        min_df=word_min_df_in,
        max_df=word_max_df_in,
        ngram_range=wordgram_range_in
    )

    # Executa a contagem de frequência das palavras na feature especificada do conjunto de dados e retorna o resultado em um formato sparse matrix
    result_w = count_vect_word.fit_transform(X[feature_objetive])

    # Se verbose for verdadeiro, exibe informações sobre o resultado da contagem de frequência das palavras
    if (verbose):
        print(str(result_w.shape))

    # Usa o objeto CountVectorizer para transformar a feature especificada do conjunto de dados em uma matriz de n-grams
    X_train_ngrams = count_vect_word.transform(X[feature_objetive])

    # Obtém uma lista com os n-grams gerados pelo objeto CountVectorizer
    list_n_grams = count_vect_word.get_feature_names_out()

    # Retorna a matriz de n-grams e a lista de n-grams gerados
    return (X_train_ngrams, list_n_grams)


def evaluate_CountVectorizer(X, y, analyzer):
    my_pipeline = Pipeline([
    ('ngrams', CountVectorizer(analyzer=analyzer)),
    ('clf', RandomForestClassifier(random_state=0))
    ])

    parameters = {}

    # ValueError: After pruning, no terms remain. Try a lower min_df or a higher max_df.
    # este erro é quando não há retornos com os filtros aplicados

    parameters.update({"CountVectorizer": {
        "ngrams__max_features": [None],
        # luego fijar: 20%, 50%, None
        "ngrams__min_df": [1],
        "ngrams__max_df": [0.5],
        "ngrams__ngram_range": [(1, 5)]
    }})

    grid_search = GridSearchCV(my_pipeline, param_grid=parameters["CountVectorizer"], cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X, y)

    print(grid_search.best_params_)

def add_params(X_train_ngrams, dataset, list_params_eval, list_n_grams):
    X_train_ngrams_aux = pd.DataFrame()

    for param in list_params_eval:
        X_train_ngrams_aux[param] = dataset[param]

    #X_train_ngrams = pd.DataFrame(X_train_ngrams.toarray(), columns=list_n_grams)
    X_train_ngrams = X_train_ngrams.join(X_train_ngrams_aux)

    list_n_grams = [*list_n_grams, *list_params_eval]
    return (X_train_ngrams, list_n_grams)


# CSR - tipo das matrizes dos ngrams
# Mais rapido que o pandas
# O resultado não é um DF pandas
def select_best_features_csr(X_train, y_train, list_ngrams):
    sel = SelectFromModel(RandomForestClassifier(n_jobs=-1,random_state=0)) #n_jobs para paralelizarlo
    sel.fit(X_train, y_train)

    support = np.asarray(sel.get_support())
    columns_with_support = list(np.array(list_ngrams)[np.array(support)])

    X_train = pd.DataFrame(X_train.toarray(), columns=list_n_grams)
    X_train_best_features = X_train[columns_with_support]

    return X_train_best_features,columns_with_support


def select_best_features_pandas(X_train, y_train, list_ngrams):
    sel = SelectFromModel(RandomForestClassifier(n_jobs=-1,random_state=0)) #n_jobs para paralelizarlo
    sel.fit(X_train, y_train)

    support = np.asarray(sel.get_support())
    columns_with_support = list(np.array(list_ngrams)[np.array(support)])

    X_train_best_features = X_train[columns_with_support]

    return X_train_best_features, columns_with_support


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

    #labels são os possiveis targets
    labels = ['Positive', 'Negative', 'Neutral']


    for cls_index in range(len(clss)):
        time_init = time.time()
        print(clss[cls_index].__str__())
        y_pred = cross_val_predict(clss[cls_index], X, y, cv=10) # classificação sempre com o cv 10

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
        "Precision macro: " + p_macro + "\n" +
        "Precision neg: " + p_micro_neg + "\n" +
        "Precision neu: " + p_micro_neu + "\n" +
        "Precicion pos: " + p_micro_pos + "\n" +
        "Recall macro: " + r_macro + "\n" +
        "Recall neg: " + r_micro_neg + "\n" +
        "Recall neu: " + r_micro_neu + "\n" +
        "Recall pos: " + r_micro_pos + "\n" +
        "F1 macro: " + f_macro + "\n" +
        "F1 neg: " + f_micro_neg + "\n" +
        "F1 neu: " + f_micro_neu + "\n" +
        "F1 pos: " + f_micro_pos + "\n" +
        "Time: " + time_total)

        print(results)

def hyperparameter_optimisation(X_train, y_train):
    parameters = {}
    parameters.update({"Random Forest": {
        "classifier__n_estimators": [100, 150],
        "classifier__class_weight": ["balanced", None],
        "classifier__max_features": ["sqrt"],
        "classifier__max_depth": [100, None],
        "classifier__criterion": ["gini"],
        "classifier__random_state": [0] # só existe nos classificadores randomizados, então ele usa o mesmo valor sempre. Mesmo em produção podemos manter o mesmo valor.
    }})

    steps = [("classifier", RandomForestClassifier())]
    pipeline = Pipeline(steps=steps)
    param_grid = parameters["Random Forest"]

    # queremos ver primeiro o scoring como accuracy
    # entretando podemos mudar isso para a precision e recall
    gscv = GridSearchCV(pipeline, param_grid, n_jobs=-1, scoring="accuracy", verbose=True, cv=5) # parametros sempre com o cv 5

    gscv.fit(X_train, y_train)
    # Get best parameters and score
    best_params = gscv.best_params_

    print("best_params")
    print(best_params)


if __name__ == "__main__":

    # lendo o arquivo csv que contém os tweets e criando um dataframe
    df = pd.read_csv('./outputs/twitter.csv', header=0)


    print(df.shape)
    print(df.columns)
    # Analisamos os resultados manualmente
    print(df.value_counts("sentiment"))

    y = df["sentiment"]
    X = df["tweet_cleaned"]

    # Aplicamos a função generate_word_ngrams no dataframe com range de 1,5
    X_train_ngrams, list_n_grams = generate_word_ngrams(X=df,
                                                        word_max_features_in=None,
                                                        word_min_df_in=1,
                                                        word_max_df_in=0.5,
                                                        wordgram_range_in=(1, 5),
                                                        verbose=False,
                                                        feature_objetive='tweet_cleaned')

    print(X_train_ngrams.shape)

    X_train_best_features, columns_with_support = select_best_features_csr(X_train_ngrams, df['sentiment'], list_n_grams)

    print(X_train_best_features.shape)

    # Adicionamos os parametros
    X_train_ngrams, list_n_grams = add_params(X_train_best_features, df, ['nouns', 'verbs', 'adjectives', 'positive', 'negative'], columns_with_support)

    X_train_best_features, columns_with_support = select_best_features_pandas(X_train_ngrams, df['sentiment'], list_n_grams)

    cross_validation(X_train_best_features, df['sentiment'])

    print(X_train_best_features.shape)
    print(len(df['sentiment']))
    evaluate_CountVectorizer(df['tweet_cleaned'] ,df['sentiment'] , 'word')

    hyperparameter_optimisation(X_train_best_features, df['sentiment'])





