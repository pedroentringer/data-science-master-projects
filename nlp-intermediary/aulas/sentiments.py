import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nlp = spacy.load('en_core_web_sm')

sia = SentimentIntensityAnalyzer()

def count_negative_words(text):
    doc = nlp(text)
    negative_count = 0

    for token in doc:
        print(token.text, sia.polarity_scores(token.text))

        if sia.polarity_scores(token.text)['neg'] > 0:
            negative_count += 1
    return negative_count

text = "I hate Mondays. The weather is terrible and my boss is mean."
count = count_negative_words(text)
print("Number of negative words:", count)