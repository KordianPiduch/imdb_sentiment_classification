import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def build_model(df, evaluate=True):
    X = df['review']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vect_param = {
        'stop_words': None,
        'max_features': None,
        'lowercase': True
    }

    sampler_param = {
        'random_state': 42,
        'n_jobs': -1
    }

    model_param = {
        'loss': 'hinge',
        'penalty': 'l2',
        'alpha': 1e-3,
        'random_state': 42,
        'max_iter': 5,
        'tol': None
    }

    pipeline_model = Pipeline(
        [
            ('vectorizer', CountVectorizer(**vect_param)),
            ('tfidf', TfidfTransformer()),
            ('sampler', SMOTE(**sampler_param)),
            ('model', SGDClassifier(**model_param))
        ]
    )

    pipeline_model.fit(X_train, y_train)

    if evaluate:
        y_pred = pipeline_model.predict(X_test)
        print('Classification report for test data set ')
        print(classification_report(y_test, y_pred))

    return pipeline_model


def save_model(path, model):
    with open(path, "wb") as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    path_processed = '../../../data/processed/imdb_processed.csv'
    path_model = "../../../models/model.pkl"

    data = pd.read_csv(path_processed, index_col=0)

    model = build_model(df=data, evaluate=False)

    save_model(path_model, model)

