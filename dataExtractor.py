#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
from pdfminer.high_level import extract_pages, extract_text
import pandas as pd
import os
import google.generativeai as genai

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
    days_left : days_left
    report_required : report_required
    num_equations : num_equations
    num_diagrams : num_diagrams
    code_required : code_required
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

data = extract_data(file)

