# importando as bibliotecas necessárias
import pandas as pd
import re
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import time
from statistics import mean
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# carregando o modelo em inglês do Spacy para realizar a análise linguística
nlp = spacy.load("en_core_web_sm")

# carregando o analisador de sentimentos do NLTK
sia = SentimentIntensityAnalyzer()

# carregando as stop words em inglês do NLTK
english_stopwords = list(stopwords.words('english'))

# função para remover as stop words (palavras comuns que não carregam significado) dos textos
def remove_stopwords(text):
    for z in english_stopwords:
        # substituindo as stop words encontradas por uma string vazia
        text = re.sub(r'\b' + z + r'\b', "", text, flags=re.IGNORECASE)
    return text


# função para realizar o pré-processamento dos textos, como remover URLs, pontuações e stop words
def preprocessing(text):
    # substituindo múltiplos espaços em branco por um único espaço
    text = re.sub(r"\s+", " ", text)

    # removendo URLs dos textos
    text = re.sub(r"(?i)\b((?:https?:\/|pic\.|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", "", text)

    # removendo as stop words dos textos
    text = remove_stopwords(text)

    # removendo espaços em branco no início e no final dos textos
    text = text.strip().rstrip()

    return text


# função para lematizar os tokens dos textos (transformar em suas formas lematizadas, ex.: "correndo" -> "correr")
def text_lemmatise(text):
    # criando um objeto do Spacy para realizar a análise linguística
    doc = nlp(text)

    # criando uma lista para armazenar as formas lematizadas dos tokens
    list_lemmas = []
    for token in doc:
        # adicionando a forma lematizada do token na lista, se ela não for uma string vazia
        list_lemmas.append(token.lemma_)
    list_lemmas = list(filter(str.strip, list_lemmas))

    # juntando as formas lematizadas dos tokens em uma única string
    sentence = ""
    for z in list_lemmas:
        sentence = sentence + " " + str(z)
        sentence = sentence.strip()

    return sentence


# Essa função recebe uma string x como entrada
def clean_text(x):
    # Pré-processa a string usando a função 'preprocessing'
    x = preprocessing(x)

    # Executa a lematização do texto usando a função 'text_lemmatise'
    x = text_lemmatise(x)

    # Retorna a string pré-processada e lematizada
    return x


# Essa função recebe uma linha de um DataFrame como entrada
def fn_counts(row):
    # Extrai o texto processado da coluna 'sentence_nlp'
    x = row['sentence_nlp']

    row['nouns'] = sum([1 for token in x if token.pos_ == 'NOUN'])
    row['verbs'] = sum([1 for token in x if token.pos_ == 'VERB'])
    row['adjectives'] = sum([1 for token in x if token.pos_ == 'ADJ'])
    row['neutral'] = sum([1 for token in x if sia.polarity_scores(token.text)['neu'] > 0])
    row['positive'] = sum([1 for token in x if sia.polarity_scores(token.text)['pos'] > 0])
    row['negative'] = sum([1 for token in x if sia.polarity_scores(token.text)['neg'] > 0])

    return row


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
    labels = ['positive', 'negative', 'neutral']


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
        "classifier__n_estimators": [100],
        "classifier__class_weight": ["balanced"],
        "classifier__max_features": ["sqrt"],
        "classifier__max_depth": [100],
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


def balance_dataset(df, target_column):
    # Conta a quantidade de ocorrências de cada classe
    class_counts = df[target_column].value_counts()

    # Encontra a menor contagem de classe
    min_count = class_counts.min()

    # Lista vazia para armazenar os dataframes balanceados
    balanced_dfs = []

    # Para cada classe, selecione aleatoriamente o número mínimo de amostras
    for sentiment, count in class_counts.iteritems():
        class_df = df[df[target_column] == sentiment].sample(min_count, random_state=42)
        balanced_dfs.append(class_df)

    # Concatena os dataframes balanceados em um único dataframe
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    # Embaralha as linhas do dataframe resultante
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df


if __name__ == "__main__":

    # lendo o arquivo csv que contém as sentenças e criando um dataframe
    df = pd.read_csv('./datasets/moodle.csv')

    df = df[:500]

    # renomeando as colunas do dataframe
    df.columns = ['sentence', 'sentiment']

    # removendo as linhas que contêm valores ausentes
    df.dropna(inplace=True, axis=0)

    # Criamos as colunas zeradas para o Pandas replicar em todas as linhas e código já entender
    df['sentence_cleaned'] = ""
    df['sentence_nlp'] = ""
    df['nouns'] = 0
    df['verbs'] = 0
    df['adjectives'] = 0
    df['neutral'] = 0
    df['positive'] = 0
    df['negative'] = 0

    # Aplicamos a função clean_text a todos as sentenças
    df['sentence_cleaned'] = df['sentence'].apply(lambda x: clean_text(x))

    # convertemos as sentenças limpos para nlp
    df['sentence_nlp'] = df['sentence_cleaned'].apply(lambda x: nlp(x))

    # contamos as informações gerais de cada sentença
    df = df.apply(lambda x: fn_counts(x), axis=1)

    df = df[['sentence', 'sentence_cleaned', 'nouns', 'verbs', 'adjectives', 'neutral', 'positive', 'negative', 'sentiment']]

    # removendo as linhas que contêm valores ausentes
    df = df.loc[df['sentence_cleaned'] != '']

    # Analisamos os resultados manualmente
    print("Contagem no dataframe original")
    print(df.value_counts("sentiment"))

    # Chamando a função para balancear o dataset
    #df = balance_dataset(df, "sentiment")

    # Exibindo as contagens das classes no dataframe balanceado
    #print("Contagem no dataframe balanceado")
    #print(df["sentiment"].value_counts())

    y = df["sentiment"]
    X = df["sentence_cleaned"]

    # Aplicamos a função generate_word_ngrams no dataframe com range de 1,5
    X_train_ngrams, list_n_grams = generate_word_ngrams(X=df,
                                                        word_max_features_in=None,
                                                        word_min_df_in=1,
                                                        word_max_df_in=0.5,
                                                        wordgram_range_in=(1, 5),
                                                        verbose=False,
                                                        feature_objetive='sentence_cleaned')

    print(X_train_ngrams.shape)

    X_train_best_features, columns_with_support = select_best_features_csr(X_train_ngrams, df['sentiment'],
                                                                           list_n_grams)

    print(X_train_best_features.shape)

    # Adicionamos os parametros
    X_train_ngrams, list_n_grams = add_params(X_train_best_features, df,
                                              ['nouns', 'verbs', 'adjectives', 'neutral', 'positive', 'negative'],
                                              columns_with_support)

    X_train_best_features, columns_with_support = select_best_features_pandas(X_train_ngrams, df['sentiment'],
                                                                              list_n_grams)

    cross_validation(X_train_best_features, df['sentiment'])

    print(X_train_best_features.shape)
    print(len(df['sentiment']))
    evaluate_CountVectorizer(df['sentence_cleaned'], df['sentiment'], 'word')

    hyperparameter_optimisation(X_train_best_features, df['sentiment'])

