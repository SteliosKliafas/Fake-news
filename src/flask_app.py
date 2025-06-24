from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
import re
import json
from src.logs import extended_logger
# from nltk.stem.porter import PorterStemmer
import torch
import os
from models import model_checkpoints_dir

port       = 8084
host       = '0.0.0.0'
verbose    = 0
pretty     = False

app = Flask(__name__)

# Load model and vectorizer

model_name = "distilbert-base-uncased"

try:
    checkpoint = torch.load(f'{model_checkpoints_dir}/distilbert-base-uncased-fine-tuned.pt',map_location=torch.device('cpu'))
except Exception as e:
    extended_logger.error(e)

try:
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2, state_dict=checkpoint['model'])
except Exception as e:
    extended_logger.error(e)
    
try:    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', state_dict=checkpoint['tokenizer'])
except Exception as e:
    extended_logger.error(e)
    
# Load the API key from the environment variable
api_key = os.environ.get('XAPI_KEY')


def predict(text):
    input_str = "<content>" +  text + "<end>"
    
    input_ids = tokenizer.encode_plus(input_str, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    
    device =  'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    
    with torch.no_grad():
        output = model(input_ids["input_ids"].to(device), attention_mask=input_ids["attention_mask"].to(device))
        
    
    return dict(zip(["Fake","Real"], [x.item() for x in list(torch.nn.Softmax()(output.logits)[0])] ))



@app.route('/predict', methods=['POST'])
def api():
    
    # Check for the presence of the X-API-Key header
    header_api_key = request.headers.get('X-Api-Key',)
    
    if header_api_key is None or header_api_key != api_key:
        # If the key is missing or incorrect, respond with a 403 Forbidden status
        extended_logger.error("403 Access forbidden. Invalid or missing X-API-Key.")
        return jsonify("Access forbidden. Invalid or missing X-API-Key."), 403
    
    else:
        # load the incoming payload. Convert it to string first
        payload = (request.data).decode('utf-8')
            # now, convert the string to json
        try:
            # When the incoming request arrived from Insomnia or Postman, the code was breaking.
            # So, check if the request first DOES NOT arrive from Insomnia or Postman
            # This line works when request comes from code, e.g. python
            payload = json.loads(payload)

        except:
            # else, if the request DOES indeed come from Insomnia or Postman,
            # Do some magic! Use different decoding and replace triple quotes with single quote
            # Then convert this temporary conversion to rawstring
            # And finally, convert rawstring to JSON using FALSE strictness
            temp_payload = (request.data).decode('unicode_escape').replace('"""', '"')
            rawstring = r"{}".format(temp_payload)
            payload = json.loads(rawstring, strict=False)


        text = payload['message']
        prediction = predict(text)
        
        return jsonify(prediction=prediction)


if __name__ == "__main__":
    app.run(host=host, port=port)