# importando as bibliotecas necessárias
import pandas as pd
import re
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords


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
    # Extrai o texto processado da coluna 'tweet_nlp'
    x = row['tweet_nlp']

    row['nouns'] = sum([1 for token in x if token.pos_ == 'NOUN'])
    row['verbs'] = sum([1 for token in x if token.pos_ == 'VERB'])
    row['adjectives'] = sum([1 for token in x if token.pos_ == 'ADJ'])
    row['positive'] = sum([1 for token in x if sia.polarity_scores(token.text)['pos'] > 0])
    row['negative'] = sum([1 for token in x if sia.polarity_scores(token.text)['neg'] > 0])

    return row


if __name__ == "__main__":

    # lendo o arquivo csv que contém os tweets e criando um dataframe
    df = pd.read_csv('./datasets/twitter_training.csv', header=None)

    # renomeando as colunas do dataframe
    df.columns = ['id', 'topic', 'sentiment', 'tweet']

    # Criamos as colunas zeradas para o Pandas replicar em todas as linhas e código já entender
    df['tweet_cleaned'] = ""
    df['tweet_nlp'] = ""
    df['nouns'] = 0
    df['verbs'] = 0
    df['adjectives'] = 0
    df['positive'] = 0
    df['negative'] = 0


    # removendo as linhas que contêm valores ausentes
    df.dropna(inplace=True, axis=0)

    # selecionando as primeiras 100 linhas do dataframe para fins de exemplo
    df = df[0:100]

    # Aplicamos a função clean_text a todos os tweets
    df['tweet_cleaned'] = df['tweet'].apply(lambda x: clean_text(x))

    # convertemos os tweets limpos para nlp
    df['tweet_nlp'] = df['tweet_cleaned'].apply(lambda x: nlp(x))

    # contamos as informações gerais de cada tweet
    df = df.apply(lambda x: fn_counts(x), axis=1)

    df = df[['id', 'tweet', 'tweet_cleaned', 'nouns', 'verbs', 'adjectives', 'positive', 'negative', 'sentiment']]

    # removendo as linhas que contêm valores ausentes
    df = df.loc[df['tweet_cleaned'] != '']

    # Salvamos para usar o conteudo processado e não fazer novamente
    df.to_csv("./outputs/twitter2.csv", index=False)


