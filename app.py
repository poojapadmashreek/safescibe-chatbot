# from flask import Flask, render_template, request, jsonify


# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch


# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


# app = Flask(__name__)

# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     return get_Chat_response(input)


# def get_Chat_response(text):

#     # Let's chat for 5 lines
#     for step in range(5):
#         # encode the new user input, add the eos_token and return a tensor in Pytorch
#         new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

#         # append the new user input tokens to the chat history
#         bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

#         # generated a response while limiting the total chat history to 1000 tokens, 
#         chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

#         # pretty print last ouput tokens from bot
#         return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


# if __name__ == '__main__':
#     app.run()

### review code 1
# from flask import Flask, render_template, request
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import pdfplumber

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# app = Flask(__name__)

# # Load content from the PDF file
# def load_content_from_pdf(file_path):
#     with pdfplumber.open(file_path) as pdf:
#         text = ""
#         for page in pdf.pages:
#             text += page.extract_text()
#     return text

# # Define the PDF file path from which you want to extract content
# pdf_file_path = r"C:/Users/Pooja Padmashree K/Downloads/QnA.pdf"  # Update this with your actual PDF file path

# # Load content from the PDF file
# content = load_content_from_pdf(pdf_file_path)

# @app.route("/")
# def index():
#     return render_template('chat.html')

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     return get_Chat_response(input)

# def get_Chat_response(text):
#     # Use the content loaded from the PDF file instead of the input text directly
#     content_to_use = content
    
#     # Let's chat for 5 lines
#     for step in range(5):
#         # encode the new user input, add the eos_token and return a tensor in Pytorch
#         new_user_input_ids = tokenizer.encode(str(content_to_use) + tokenizer.eos_token, return_tensors='pt')

#         # append the new user input tokens to the chat history
#         bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

#         # generated a response while limiting the total chat history to 1000 tokens, 
#         chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

#         # pretty print last output tokens from bot
#         return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

# if __name__ == '__main__':
#     app.run()

### review code 2

# from flask import Flask, render_template
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# app = Flask(__name__)

# # Load the pre-trained model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# # Define the question and answer pair
# question = "Is it normal to have periods twice a month?"
# answer = ("For menstrual cycles that are 21 days long, it’s possible to have a period twice "
#           "during one calendar month. However, bleeding twice or more during one menstrual "
#           "cycle, blood after sex, or bleeding that stops and then starts again after several days "
#           "could indicate a health issue.")

# def get_chat_response():
#     # Let's chat for 5 lines
#     for step in range(5):
#         # encode the input text, add the eos_token and return a tensor in Pytorch
#         input_text = question + " " + answer  # Combine question and answer
#         input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

#         # generate a response while limiting the total chat history to 1000 tokens
#         chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

#         # pretty print last output tokens from bot
#         return tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

# @app.route("/")
# def index():
#     return render_template('chat.html')

# @app.route("/get")
# def chat():
#     return get_chat_response()

# if __name__ == '__main__':
#     app.run()

###review code 3


import os
import requests
from flask import Flask, render_template, request
import pdfplumber

app = Flask(__name__)

# Load the ChatGPT API key from environment variable
api_key = os.getenv("CHATGPT_API_KEY")

# Define the PDF file path
pdf_file_path = "C:/Users/Pooja Padmashree K/Downloads/QnA.pdf"

# Load content from the PDF file
def load_content_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Load content from the PDF file
content = load_content_from_pdf(pdf_file_path)

def get_chat_response(msg):
    try:
        # Prepare data for the API request
        data = {
            "model": "davinci-codex",
            "prompt": content + msg,
            "max_tokens": 150
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer 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"

        }

        # Make API request to generate response
        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()

        # Extract and return the response text
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        return f"Error occurred: {e}"

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = get_chat_response(msg)
    return response

if __name__ == '__main__':
    app.run()
