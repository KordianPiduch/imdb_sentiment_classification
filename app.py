import pickle
from fastapi import FastAPI, HTTPException
from src.sentiment_classification.preprocess_data.process_data import process_sentence

# load model
with open('./models/model.pkl', 'rb') as file:
    model = pickle.load(file)

# initialize an instance of FastAPI
app = FastAPI()


@app.get("/")
def root():
    return {
        'project': 'IMDB Review Sentiment Classification',
        'author': 'Kordian Piduch'
    }


@app.post("/predict_sentiment")
def predict_sentiment(text_message):
    if not text_message:
        raise HTTPException(status_code=400,
                            detail="Please Provide a valid text message")

    # preprocess received sentenced
    sentence = process_sentence(text_message)

    prediction = model.predict([sentence])

    return {
        "text_message": text_message,
        "sentiment": prediction[0]
    }
