# IMDB Reviews - Sentiment classification 

**Python version:** 3.10.  
**Packages:** pandas, numpy, sklearn, imblearn, matplotlib, fastapi, uvicorn, nltk, bs4.

# About dataset
Dataset used in this project is modified imdb dataset that can be found on kaggle.    

![alt text](https://github.com/KordianPiduch/imdb_sentiment_classification/blob/main/reports/data_imbalanced.jpg "imbalanced")

Dataset is very imbalanced, and reviews need to be cleaned before applying to model. 
All steps are shown in /notebooks/1-kp-process-data.ipynb

# Classification result
Choosing and comapring models process can be found in /notebooks/2-kp-build-model.ipynb.  

Oversampling minority class using SMOTE

Selected model: Linear SVM implemented using SGDClassifier

Prediction results on test data:
```
              precision    recall  f1-score   support

    negative       0.77      0.79      0.78       473
    positive       0.98      0.98      0.98      5028

    accuracy                           0.96      5501
   macro avg       0.88      0.88      0.88      5501
weighted avg       0.96      0.96      0.96      5501

```
Confusion Matrix:  
![alt text](https://github.com/KordianPiduch/imdb_sentiment_classification/blob/main/reports/model_ConfusionMatrix.jpg)


# Run application without docker 

### 1. Enter project root folder
```bash
cd imdb_sentiment_classification
```

### 2. Create and activate new virtual environent

Linux / MacOS: 
```bash
python3 -m venv venv

source venv/bin/activate
```

Windows Command Prompt:
```bash
python -m venv venv

venv\Scripts\activate.bat
```

### 3. Install required packages inside created environment
```bash
pip install -e .
```

### 4. Run script to preprocess, train and save model
Linux / MacOS:
```bash
python3 train.py
```

Windows:
```bash
python train.py
```

### 5. Run uvicorn server to get access to api.
```bash
uvicorn app:app
```

### 6. Enter address in web broswer to get information about project and author:
```
http://127.0.0.1:8000
```

### 7. Test API:
Steps how to test API are described below.


# Run application with docker image
All infomarion about installation and setup docker process can be found here: [Docker documentation](https://docs.docker.com)

### 1. Enter project root folder
```bash
cd imdb_sentiment_classification
``` 

### 2. Create image
```bash
docker build -t imdb_project .
```
### 3. Run image
```bash
docker run -p 8000:8000 imdb_project
```
### 4. Enter address in web broswer to get information about project and author:
```
http://127.0.0.1:8000
```
### 5. Test API:
Steps how to test API are described below.


# Test API
When docker image is running (or uvicorn server is active). 

Open main page by entering addres below in broswer.
```
http://127.0.0.1:8000
```
Main page contans project name and author.

### Perform POST request using SwaggerUI
API can be tested with SwaggerUI by going to the **/docs** path extension and testing a sample POST request.
```
http://127.0.0.1:8000/docs
```
##### How to send reguest with review for sentiment classicifation
1. click on POST request 
2. press "Try it out"
3. type your review in "text_messege" field
4. press "Execute"

Below in responses API return text_message and sentiment in JSON format

```
{
  "text_message": "some super cool and great review",
  "sentiment": "positive"
}
```
