from flask import Flask, render_template, request, redirect, url_for,session
from authlib.integrations.flask_client import OAuth
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from pdfminer.high_level import extract_pages, extract_text
import os
import re
import pandas as pd
from io import BytesIO
import google.generativeai as genai
app = Flask(__name__)

app.secret_key = 'super_secret_key'

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    pass

@login_manager.user_loader
def load_user(user_id):
    # This function should return a User object based on user_id.
    user = User()
    user.id = user_id  # In your case, the user_id could be the Auth0 user ID
    return user

# Set up a folder for uploaded files
UPLOAD_FOLDER = 'uploads'  # You can change this to a different directory if needed
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}  # Allowed file type is PDF

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/index.html')
@login_required
def index():
    return render_template('index.html')

def extract_data(pdf):
    assignment_text = extract_text(pdf)
    
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
    Assignment description: "{assignment_text}"
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
file_bytes = None
@app.route('/upload', methods=['POST'])
def upload_file():
    global file_bytes  # Use the global keyword to modify the global variable

    # Check if file is in the request
    if 'file' not in request.files:
        return render_template('index.html', message="No file selected.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message="No file selected.")

    # Check if the file is allowed and save it
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("helloooo")
        file_bytes = BytesIO(file.read()) 
        print("iofile", file_bytes)
        extracted_text = extract_text(file_bytes)  # pdfminer extracts text from the PDF

        print("Extracted text from PDF: ", extracted_text)
        return render_template('index.html', message=f"File uploaded successfully: {filename}")
    else:
        # Invalid file format
        return render_template('index.html', message="Invalid file format. Please upload a PDF file.")
    

@app.route('/extract_data', methods=['POST'])
def extract_assignment_data():
    print("file_bytes", file_bytes)
    if file_bytes is None:
        return render_template('index.html', message="No file uploaded.")
    print("B ", file_bytes)
    data = extract_data(file_bytes)
    print("data1 ", data)
    #return render_template('index.html', message="Data extracted successfully.")

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)