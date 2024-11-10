from flask import Flask, render_template, request, redirect, url_for
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.high_level import extract_text
import os
import re
import pandas as pd
from io import BytesIO
import google.generativeai as genai
import torch
import numpy
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        # Define the layers
        self.hidden = nn.Linear(input_size, hidden_size)   # Hidden layer
        self.output = nn.Linear(hidden_size, output_size)  # Output layer (linear activation)

    def forward(self, x):
        # Forward pass: input -> hidden layer with ReLU -> output layer (no activation)
        x = torch.relu(self.hidden(x))      # Hidden layer with ReLU activation
        x = self.output(x)                  # Linear output layer
        return x
    
app = Flask(__name__)

# Set up a folder for uploaded files
UPLOAD_FOLDER = 'uploads'  # You can change this to a different directory if needed
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}  # Allowed file type is PDF

# Function to check allowed file extensions

def allowed_file(filename):
    """Check if the file extension is PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'
#def allowed_file(filename):
    #return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    # Render the page without any message initially
    return render_template('index.html')

def extract_data(pdf):
    #assignment_text = extract_text(pdf)
    
    genai.configure(api_key="AIzaSyAcZ_8woXZrFZ7pLZa41y8oaTre9j2EV9Y")
    generation_config = {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 40,
      "max_output_tokens": 8192,
      "response_mime_type": "text/plain",
    }
    
    model = genai.GenerativeModel(
      model_name="gemini-1.5-pro-002",
      generation_config=generation_config,
    )
    
    chat_session = model.start_chat(
      history=[
      ]
    )
    
    prompt = f"""
    Given the following assignment description, extract and summarize the relevant details in a structured table. Please identify the following attributes:
    
    The number of questions or problems in the assignment (num_questions).
    The deadline or timeline for the assignment (days_left). Calculate the number of days from the 10th november 2024 to the submission deadline.
    Whether the assignment requires a report (report_required). Output 1 if a report is required, otherwise output 0.
    The number of equations in the assignment (num_equations).
    The number of diagrams required to be drawn (num_diagrams).
    Whether coding is required for the assignment (code_required). Output 1 if coding is required, otherwise output 0.
    Then, output the information in a dictionary with the keys as follows:
    num_questions : num_questions
    num_equations : num_equations
    num_diagrams : num_diagrams
    report_required : report_required
    code_required : code_required
    days_left : days_left
    Assignment description: "{pdf}"#assignment_text
    """
    
    response = chat_session.send_message(prompt)   
    extracted_info = response.text  
    #print(extracted_info)
    dict_pattern = r"\{(.*?)\}"
    dict_match = re.search(dict_pattern, extracted_info, re.DOTALL)
    if dict_match:
        dict_str = dict_match.group(0)
        dict_str = dict_str.replace("'", '"')
        extracted_dict = eval(dict_str)
        #print(extracted_dict)
        # df = pd.DataFrame(list(extracted_dict.items()), columns=['Attribute', 'Value'])
        df = pd.DataFrame([extracted_dict])
    return df 

# Declare file as a global variable
extracted_text = None
data = None
@app.route('/upload', methods=['POST'])
def upload_file():
    # Use a global variable for file_bytes
    global extracted_text

    # Check if a file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', message="No file selected.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message="No file selected.")

    # Check if the file is a valid PDF and save it
    if file and allowed_file(file.filename):
        filename = file.filename
        file_bytes = BytesIO(file.read())  # Convert file to bytes
        print("File in bytes:", file_bytes)

        # Extract text from the PDF using pdfminer
        extracted_text = extract_text(file_bytes)
        global data
        #print("file_bytes", extracted_text)
        if extracted_text is None:
            return render_template('index.html', message="No file uploaded.")
        #print("B ", file_bytes)
        data = extract_data(extracted_text)
        print(data)
        print("type", type(data))
    
        input_size = 6           # Number of input features
        hidden_size = 64         # Number of neurons in the hidden layer
        output_size = 1          # Output size (e.g., regression task)

        model = SimpleNN(input_size, hidden_size, output_size)

        # Load the saved weights from the .pth file
        checkpoint = torch.load('model.pth')  # Replace 'model.pth' with the path to your .pth file

        model.load_state_dict(checkpoint)

        model.eval()

        input_list = data.values.tolist()
        print(input_list)
        input_tensor = torch.tensor(input_list, dtype=torch.float32)

        output_model = model(input_tensor)

        # Print the model's output (predictions)
        #print("Model output:", output/60)
        scaled_output = output_model / 60
        print("Model output:", scaled_output)
        extracted_output = round(scaled_output.item())
        number = int(extracted_output)

        # Check if the number is greater than 20
        if number > 20:
            diff = "hard"
        elif number >12:
            diff = "med"
        else:
            diff = "easy"

        return render_template(
            'index.html', 
            message=f"File uploaded successfully: {filename}",
            extracted_output=str(extracted_output) + " Hours",
            diff=diff,  # Or "med" / "easy" based on model output
            auto_refresh=True  # Flag to trigger JavaScript reload
        )
        #return render_template('index.html', message=f"File uploaded successfully: {filename}",extracted_output=str(extracted_output) + " Hours", diff = diff)
    else:
        # Invalid file format
        return render_template('index.html', message="Invalid file format. Please upload a PDF file.")


if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)