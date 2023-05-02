import os
import logging
from flask import Flask, request, jsonify
from ultralytics import YOLO
#from model.model import EnFrTranslator


app = Flask(__name__)  

# define model path
model_path = './model/best.pt'


# create instance
model = YOLO(model_path)
logging.basicConfig(level=logging.INFO)


#creating the different routes to be used
@app.route('/')
def home():
    return  render_template('index.html', pred = 'Welcome to FASMAR\'s English to French translator! Please type an English sentence above.')

@app.route('/segment', methods=['POST'])
def translate():
  #getting input from the user (video)
  input = cap = cv2.VideoCapture()

  #input fromm user (pic)

  #preprocessing user input to match model parameters


  #obtaining model prediction
  results = model.predict(source=[frame], conf=0.70, save=False)


  return render_template('index.html' , pred = 'The sentence in French: {}'.format(pred))

def main():
    """Run the Flask app."""
    app.run(host="0.0.0.0", port=8000, debug=False)

#running the application
if __name__ == "__main__":
    main()
