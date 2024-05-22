from flask import Flask, json,request,jsonify
import numpy as np
import sklearn
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.preprocessing.text import tokenizer_from_json # type: ignore
import tensorflow
import keras
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))    
lemmatizer = WordNetLemmatizer()


# importing model
model=keras.models.load_model('model.h5')
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = json.load(f)
    tokenizer_json_string = json.dumps(tokenizer_json)  # Convert dictionary to JSON string
    tokenizer = tokenizer_from_json(tokenizer_json_string)


# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "Machine Learning Based Classification of Suicidal Thoughts from Suicidal Comments"

@app.route("/predict",methods=['POST'])
def predict():
    # Define the text
    text =request.form.get('text')

    text=text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize the text
    text = word_tokenize(text)

    # Remove stopwords
    text = [word for word in text if word not in stop_words]

    # Lemmatize the words
    text = [lemmatizer.lemmatize(word) for word in text]

    # Join
    text = ' '.join(text)
    sequences = tokenizer.texts_to_sequences([text])

    padded_sequences = pad_sequences(sequences, maxlen=300, padding="post", truncating="post")
    # get predictions for toxicity
    predictions = model.predict(padded_sequences)[0]

    # Format prediction text
    prediction_dict = {}
    for i, col in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
        prediction_dict[col]=float(predictions[i])

    result=jsonify(prediction_dict)
  
    return result


# python main
if __name__ == "__main__":
    app.run(debug=True)