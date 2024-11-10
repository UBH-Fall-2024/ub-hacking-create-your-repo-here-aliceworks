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
# login_manager = LoginManager(app)
login_manager = LoginManager()
login_manager.init_app(app)
users={'saloni1829@gmail.com':generate_password_hash('password')}

class User(UserMixin):
    pass

@login_manager.user_loader
def load_user(user_id):
    return User()

oauth = OAuth(app)
auth0 = oauth.register(
    "auth0",
    client_id='daOUnbPRXMOeCFAiIV46Gv3eUe8bUHvQ',
    client_secret='tLzyu_JUCix-WJVyTKrnkFOKtYlKsxv63g-8bMl0v2yQPfeNPYOqGb9hy9L6_XIb',
    access_token_url = 'https://dev-pwcvgus2o8tbmu2g.us.auth0.com/oauth/token',
    authorize_url = 'https://dev-pwcvgus2o8tbmu2g.us.auth0.com/authorize',
    client_kwargs={
        "scope": "openid profile email",
    },
)
@app.route('/')
def home():
    return render_template("home.html", user=session.get("users"))

@app.route('/login')
def login():
    return auth0.authorize_redirect(redirect_url='http://127.0.0.1:5000/callback',_external=True)


@app.route('/callback')
def callback():
    try:
        resp = auth0.authorize_access_token()
        print("Response from Auth0:", resp)  # Debugging line
        session['jwt_payload'] = resp.json()
        user = User()
        user.id = session['jwt_payload']['sub']
        login_user(user)
        return redirect('http://127.0.0.1:5000/index.html')
    except Exception as e:
        print(f"Error during callback: {e}")  # Debugging line
        return redirect('/error')

@app.route('/login_direct', methods=["GET", "POST"])
def login_direct():
    if check_password_hash(users.get(request.form['username'], ''), request.form['password']):
        user=User()
        user.id=request.form['username']
        login_user(user)
        return redirect('http://127.0.0.1:5000/index.html')
    else:
        session['error']= 'Invalid credentials, please try again!'
        return redirect('/')
    
@app.route('/index.html')
@login_required
def index():
    return redirect('http://127.0.0.1:5000/index.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect('/login')


if __name__ == "__main__":
    app.run(debug=True)
