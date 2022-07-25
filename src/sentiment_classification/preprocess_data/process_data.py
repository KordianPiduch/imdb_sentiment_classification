import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup

nltk.download('stopwords')
nltk.download('punkt')

custom_stop_words = nltk.corpus.stopwords.words('english')
try:
    custom_stop_words.remove('from')
except ValueError:
    pass


def remove_html_tags(sentence: str) -> str:
    soup = BeautifulSoup(sentence, 'html.parser')
    return soup.get_text()


def remove_non_alphanumeric(sentence: str) -> str:
    # remove nonalpha-numeric characters
    sentence = re.sub(r'[^a-zA-Z\d\s]+', ' ', sentence)
    # replace multiple spaces with one space
    sentence = re.sub(r' +', ' ', sentence)
    return sentence


def remove_stop_words(sentence: str, stop_words: list) -> list:
    word_tokens = nltk.tokenize.word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return filtered_sentence


def remove_standalone_numbers(sentence: list) -> list:
    filtered_sentence = [w for w in sentence if not w.isdigit()]
    return filtered_sentence


def sentence_stemmer(sentence: list) -> list:
    porter = nltk.stem.PorterStemmer()
    stemmed_sentence = [porter.stem(w) for w in sentence]
    return stemmed_sentence


def process_sentence(sentence: str) -> str:
    sentence = remove_html_tags(sentence)
    sentence = remove_non_alphanumeric(sentence)
    sentence = remove_stop_words(sentence, custom_stop_words)
    sentence = remove_standalone_numbers(sentence)
    sentence = sentence_stemmer(sentence)
    return ' '.join(sentence)


def process_dataset(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df['review'] = df['review'].apply(process_sentence)
    return df


if __name__ == '__main__':
    raw = '../../../data/raw/IMDB Dataset.csv'
    processed = '../../../data/processed/imdb_processed.csv'

    data = process_dataset(raw)
    data.to_csv(processed)
