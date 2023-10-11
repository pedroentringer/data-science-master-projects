import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re
import spacy
import warnings

warnings.filterwarnings("ignore")

# carregando o modelo em inglês do Spacy para realizar a análise linguística
nlp = spacy.load("en_core_web_sm")

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

    # removendo numeros
    text = re.sub(r'\d+', '', text)

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
    return x.lower()

def generate_word_gram(df, sentiment_target):
    filtered_df = df.loc[df['sentiment'] == sentiment_target]

    # Cria um objeto CountVectorizer com os parâmetros especificados
    count_vect_word = CountVectorizer(
        analyzer='word',
        lowercase=True,
        ngram_range=(2, 2)
    )

    count_vect_word.fit_transform(filtered_df['sentence'])

    # Obtém uma lista com os n-grams gerados pelo objeto CountVectorizer
    list_n_grams = count_vect_word.get_feature_names_out()

    return list_n_grams


def filter_unique_elements(main_list, compare_list1, compare_list2):
    repeated_elements = set(compare_list1).union(set(compare_list2))
    filtered_list = [item for item in main_list if item not in repeated_elements]

    return filtered_list


def convert_to_regex(ngram):
    regex = ''.join([r'\b' + item + r'\b|' for item in ngram])
    return regex[:-1]


def has_match(regex, sentence):
    matches = re.finditer(regex, sentence, re.MULTILINE)
    return len(list(enumerate(matches))) > 0


def run_classification(line, neutral_regex, positive_regex, negative_regex):
    classification = -1
    sentence = line['sentence']
    target = line['sentiment']

    if has_match(neutral_regex, sentence):
        classification = 'neutral'
    elif has_match(positive_regex, sentence):
        classification = 'positive'
    elif has_match(negative_regex, sentence):
        classification = 'negative'

    if classification != target and classification != -1:
        print("\n\nCONFLITO")
        print(sentence)
        print("Original ", target)
        print("Classificação ", classification)

    line['dictionary_classification'] = classification

    return line


if __name__ == "__main__":
    # lendo o arquivo csv que contém as sentenças e criando um dataframe
    df = pd.read_csv('./datasets/moodle.csv')

    # renomeando as colunas do dataframe
    df.columns = ['sentence', 'sentiment']

    df['sentence'] = df['sentence'].apply(lambda x: clean_text(x))

    # removendo as linhas que contêm valores ausentes
    df.dropna(inplace=True, axis=0)

    neutral_ngram = generate_word_gram(df, 'neutral')
    positive_ngram = generate_word_gram(df, 'positive')
    negative_ngram = generate_word_gram(df, 'negative')

    print("\n")
    print("Original")
    print("neutral: ", len(neutral_ngram))
    print("positive: ", len(positive_ngram))
    print("negative: ", len(negative_ngram))

    filtered_neutral_ngram = filter_unique_elements(neutral_ngram, positive_ngram, negative_ngram)
    filtered_positive_ngram = filter_unique_elements(positive_ngram, neutral_ngram, negative_ngram)
    filtered_negative_ngram = filter_unique_elements(negative_ngram, positive_ngram, neutral_ngram)

    print("\n")
    print("Filtrado")
    print("neutral: ", len(filtered_neutral_ngram))
    print("positive: ", len(filtered_positive_ngram))
    print("negative: ", len(filtered_negative_ngram))

    confirm_neutral_ngram = filter_unique_elements(filtered_neutral_ngram, filtered_positive_ngram, filtered_negative_ngram)
    confirm_positive_ngram = filter_unique_elements(filtered_positive_ngram, filtered_neutral_ngram, filtered_negative_ngram)
    confirm_negative_ngram = filter_unique_elements(filtered_negative_ngram, filtered_positive_ngram, filtered_neutral_ngram)

    print("\n")
    print("Confirmação")
    print("neutral: ", len(confirm_neutral_ngram))
    print("positive: ", len(confirm_positive_ngram))
    print("negative: ", len(confirm_negative_ngram))

    neutral_regex = convert_to_regex(filtered_neutral_ngram)
    positive_regex = convert_to_regex(filtered_positive_ngram)
    negative_regex = convert_to_regex(filtered_negative_ngram)

    print("\n")
    print("Regex")
    print("neutral: ", neutral_regex)
    print("positive: ", positive_regex)
    print("negative: ", negative_regex)

    print("\n")
    df = df.apply(lambda x: run_classification(x, neutral_regex, positive_regex, negative_regex), axis=1)

    print("\n")
    print("Contagem da classificação")
    print(df['dictionary_classification'].value_counts())

    df.to_csv('./outputs/final.csv', index=False, header=True)



