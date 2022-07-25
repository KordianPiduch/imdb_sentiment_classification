import warnings
from sentiment_classification.preprocess_data.process_data import process_dataset
from sentiment_classification.train_model.build_model import build_model, save_model

warnings.filterwarnings("ignore")

PATH_TO_MODEL = './models/model.pkl'
PATH_TO_DATA_RAW = './data/raw/IMDB Dataset.csv'

if __name__ == '__main__':
    df = process_dataset(PATH_TO_DATA_RAW)
    model = build_model(df=df, evaluate=False)
    save_model(PATH_TO_MODEL, model)
