# Import necessary libraries
from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_cors import CORS
from flask_login import login_required, current_user, login_user, logout_user, LoginManager, UserMixin
from flask.logging import create_logger
from flask_mysqldb import MySQL
from functools import wraps
import os
import secrets
import validators
import pickle
import urllib
from langdetect import detect
from newspaper import Article, Config
from newsapi import NewsApiClient
from werkzeug.security import generate_password_hash, check_password_hash
from utils import save_history, is_valid_url 
from config import Config as AppConfig  
import pandas as pd  
import newspaper  
import numpy as np  
from datetime import datetime  
import torch
from model import ChronNet, train_model  # Import ChronNet class and training function
import threading
import schedule
import time

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)
app.config.from_object(AppConfig)

log = create_logger(app)
mysql = MySQL(app)

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  

# Define User class
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

# Define user_loader function
@login_manager.user_loader
def load_user(user_id):
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
    account = cursor.fetchone()
    cursor.close()
    if account:
        return User(id=account['id'], username=account['username'], email=account['email'])
    return None

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key=AppConfig.NEWS_API_KEY)

@app.route('/', methods=['GET', 'POST'])
def main():
    """Render the main page with top news headlines."""
    data = newsapi.get_top_headlines(language='en', country="us", category='general', page_size=10)
    l1, l2 = zip(*[(i['title'], i['url']) for i in data['articles']])
    return render_template('main.html', l1=l1, l2=l2)

@app.route('/login')
def login():
    """Render the login page."""
    registered = request.args.get('registered')  # Check if redirected from registration
    if registered:
        flash('Registration successful! You can now log in.', 'login_success')
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    """Handle login form submission for both users and admin."""
    email = request.form.get('email')
    password = request.form.get('password')

    if current_user.is_authenticated:
        return redirect('/history')

    cursor = None  # Initialize cursor to None
    try:
        # Check if the login is for admin
        if email == "admin@fakenews.com" and password == "admin123":
            session.clear()  # Clear any existing session
            session['admin_logged_in'] = True
            flash('Admin login successful!', 'login_success')
            return redirect(url_for('admin_dashboard'))

        # Check if the login is for a regular user
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        account = cursor.fetchone()

        if account:
            if account['banned']:
                flash('Your account has been banned. Please contact support.', 'login_error')
                return redirect(url_for('login'))
            password_db = account['password_hash']
            if check_password_hash(password_db, password):
                user = User(id=account['id'], username=account['username'], email=account['email'])
                login_user(user)
                session['logged_in'] = True
                session['username'] = account['username']
                session['id'] = account['id']
                flash('You have successfully logged in!', 'login_success')
                return redirect(url_for('main'))
            else:
                flash('Incorrect password. Please try again.', 'login_error')
        else:
            flash('No account found with this email address.', 'login_error')
    except Exception as e:
        flash(f'Error: {e}', 'login_error')
    finally:
        if cursor:  # Close the cursor only if it was initialized
            cursor.close()

    return render_template('login.html', email=email)

@app.route('/register', methods=['POST', 'GET'])
def register():
    """Handle registration form submission."""
    email = request.form.get('email')
    username = request.form.get('username')
    password = request.form.get('password')

    if request.method == 'POST':
        try:
            cursor = mysql.connection.cursor()
            cursor.execute('SELECT * FROM users WHERE email LIKE %s', (email,))
            account = cursor.fetchone()

            if account:
                flash('An account with this email already exists.', 'register_error')
            elif len(password) < 8:
                flash('Password must be at least 8 characters long.', 'register_error')
            elif not username or not password or not email:
                flash('All fields are required.', 'register_error')
            else:
                password_hash = generate_password_hash(password)
                cursor.execute("INSERT INTO users(email, username, password_hash) VALUES(%s, %s, %s)", (email, username, password_hash))
                mysql.connection.commit()
                flash('Registration successful! You can now log in.', 'register_success')
                return redirect(url_for('login', registered=True))  # Pass a query parameter
        except Exception as e:
            flash(f'Error: {e}', 'register_error')
        finally:
            cursor.close()

    return render_template('register.html', email=email, username=username)

# Check if user logged in
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Please login to gain access of this page', 'access_error')
            return redirect(url_for('login'))
    return wrap

@app.route('/logout')
def logout():
    session.clear()
    logout_user()
    return redirect('/')

@app.route('/history', methods=['GET', 'POST'])
@is_logged_in
def history():
    userID = session['id']
    cursor = mysql.connection.cursor()
    result = cursor.execute('SELECT * FROM history WHERE userID = %s ORDER BY historyDate DESC', (userID,))
    history = cursor.fetchall()
    cursor.close()

    if history:
        record = True
        return render_template('history.html', history=history, record=record)
    else:
        msg = 'No History Found'
        return render_template('history.html', msg=msg, record=False)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict whether the news is true or false using ChronNet."""
    url = request.form.get('news')  # Get the URL from the form input
    print(f"Received URL: {url}") 

    if is_valid_url(url):
        try:
            article = Article(str(url))
            article.download()
            article.parse()
            parsed = article.text
            print(f"Parsed article text: {parsed[:100]}...")  

            if parsed:
                lang = detect(parsed)
                print(f"Detected language: {lang}") 

                if lang == "en":
                    article.nlp()
                    news = article.text

                    if news:
                        # Load vectorizer and ChronNet model
                        try:
                            vectorizer = pickle.load(open('TfidfVectorizer-ChronNet.sav', 'rb'))
                            model = ChronNet(input_dim=vectorizer.transform([news]).shape[1], hidden_dim=128, output_dim=2)
                            model.load_state_dict(torch.load('ChronNetModel.pth'))
                            model.eval()
                        except FileNotFoundError as e:
                            print(f"Error: {e}")  
                            flash('Model file is missing. Please train the model first.', 'predict_error')
                            return redirect(url_for('main'))

                        X_input = vectorizer.transform([news]).toarray()
                        X_tensor = torch.tensor(X_input, dtype=torch.float32)

                        if len(X_tensor.shape) == 2:
                            X_tensor = X_tensor.unsqueeze(0)  

                        # Predict
                        with torch.no_grad():
                            outputs = model(X_tensor)
                            _, predicted = torch.max(outputs, 1)
                            outcome = "Real" if predicted.item() == 1 else "Fake"
                            print(f"Prediction outcome: {outcome}")  

                        prediction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        if 'logged_in' in session:
                            userID = session['id']
                            save_history(mysql, userID, url, outcome)

                        # Render the result page
                        return render_template(
                            'predict.html',
                            prediction_text=outcome,
                            true_percentage=outputs[0][1].item() * 100,
                            false_percentage=outputs[0][0].item() * 100,
                            url_input=url,
                            news=news,
                            prediction_date=prediction_date
                        )
                    else:
                        flash('Invalid URL! Please try again', 'predict_error')
                        return redirect(url_for('main'))
                else:
                    flash('We currently do not support this language', 'predict_error')
                    return redirect(url_for('main'))
            else:
                flash('Invalid news article! Please try again', 'predict_error')
                return redirect(url_for('main'))
        except Exception as e:
            print(f"Error during prediction: {e}")  # Debugging log
            flash(f'Error: {e}', 'predict_error')
            return redirect(url_for('main'))
    else:
        flash('Please enter a valid news site URL', 'predict_error')
        return redirect(url_for('main'))

@app.route('/community', methods=['GET', 'POST'])
@is_logged_in
def community():
    """Community page for users to post, like, and reply."""
    if request.method == 'POST':
        post_text = request.form.get('post')
        reply_to = request.form.get('reply_to')  # Optional: ID of the post being replied to
        user_id = session['id']

        # Word filtering
        banned_words = ['spam', 'offensive', 'abuse']
        if any(word in post_text.lower() for word in banned_words):
            flash('Your post contains inappropriate content.', 'community_error')
            return redirect(url_for('community'))

        cursor = mysql.connection.cursor()
        if reply_to:
            # Add a reply to an existing post
            cursor.execute("INSERT INTO replies(postID, userID, replyText) VALUES(%s, %s, %s)", (reply_to, user_id, post_text))
        else:
            # Add a new post
            cursor.execute("INSERT INTO posts(userID, postText) VALUES(%s, %s)", (user_id, post_text))
        mysql.connection.commit()
        cursor.close()
        flash('Your post has been added successfully!', 'community_success')
        return redirect(url_for('community'))

    # Fetch posts and replies
    cursor = mysql.connection.cursor()
    cursor.execute("""
        SELECT p.id AS postID, p.postText, p.createdAt, u.username, 
               (SELECT COUNT(*) FROM likes WHERE likes.postID = p.id) AS likeCount
        FROM posts p
        JOIN users u ON p.userID = u.id
        ORDER BY p.createdAt DESC
    """)
    posts = cursor.fetchall()

    cursor.execute("""
        SELECT r.postID, r.replyText, r.createdAt, u.username
        FROM replies r
        JOIN users u ON r.userID = u.id
        ORDER BY r.createdAt ASC
    """)
    replies = cursor.fetchall()
    cursor.close()

    return render_template('community.html', posts=posts, replies=replies)

@app.route('/like/<int:post_id>', methods=['POST'])
@is_logged_in
def like_post(post_id):
    """Allow users to like a post only once."""
    user_id = session['id']
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM likes WHERE postID = %s AND userID = %s", (post_id, user_id))
    like = cursor.fetchone()
    if like:
        flash('You have already liked this post.', 'community_error')
    else:
        cursor.execute("INSERT INTO likes(postID, userID) VALUES(%s, %s)", (post_id, user_id))
        mysql.connection.commit()
        flash('You liked the post!', 'community_success')
    cursor.close()
    return redirect(url_for('community'))

@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    """Admin login page."""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Check admin credentials
        if email == "admin@fakenews.com" and password == "admin123":
            session.clear()  # Clear any existing session
            session['admin_logged_in'] = True
            flash('Admin login successful!', 'admin_success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials. Please try again.', 'admin_error')

    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    """Admin dashboard."""
    if not session.get('admin_logged_in'):
        flash('Please log in as admin to access this page.', 'admin_error')
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor()
    cursor.execute('SELECT * FROM users WHERE banned = 0')  # Fetch active users
    users = cursor.fetchall()
    cursor.execute('SELECT * FROM users WHERE banned = 1')  # Fetch banned users
    banned_users = cursor.fetchall()
    cursor.execute('SELECT * FROM comments')
    comments = cursor.fetchall()
    cursor.execute("""
        SELECT p.id AS postID, p.postText, u.username
        FROM posts p
        JOIN users u ON p.userID = u.id
        ORDER BY p.createdAt DESC
    """)
    posts = cursor.fetchall()
    cursor.execute("""
        SELECT r.id, r.replyText, u.username
        FROM replies r
        JOIN users u ON r.userID = u.id
        ORDER BY r.createdAt ASC
    """)
    replies = cursor.fetchall()
    cursor.close()

    return render_template('admin_dashboard.html', users=users, banned_users=banned_users, comments=comments, posts=posts, replies=replies)

@app.route('/admin/remove_user/<int:user_id>', methods=['POST'])
def remove_user(user_id):
    """Remove a user."""
    if not session.get('admin_logged_in'):
        flash('Unauthorized access.', 'admin_error')
        return redirect(url_for('login'))

    try:
        cursor = mysql.connection.cursor()
        # Delete related rows in the history and comments tables
        cursor.execute('DELETE FROM history WHERE userID = %s', (user_id,))
        cursor.execute('DELETE FROM comments WHERE userID = %s', (user_id,))
        # Delete the user
        cursor.execute('DELETE FROM users WHERE id = %s', (user_id,))
        mysql.connection.commit()
        flash('User removed successfully.', 'admin_success')
    except Exception as e:
        flash(f'Error: {e}', 'admin_error')
    finally:
        cursor.close()

    return redirect(url_for('admin_dashboard'))

@app.route('/admin/remove_comment/<int:comment_id>', methods=['POST'])
def remove_comment(comment_id):
    """Remove a comment."""
    if not session.get('admin_logged_in'):
        flash('Unauthorized access.', 'admin_error')
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor()
    cursor.execute('DELETE FROM comments WHERE id = %s', (comment_id,))
    mysql.connection.commit()
    cursor.close()

    flash('Comment removed successfully.', 'admin_success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/ban_user/<int:user_id>', methods=['POST'])
def ban_user(user_id):
    """Ban a user."""
    if not session.get('admin_logged_in'):
        flash('Unauthorized access.', 'admin_error')
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor()
    cursor.execute('UPDATE users SET banned = 1 WHERE id = %s', (user_id,))
    mysql.connection.commit()
    cursor.close()

    flash('User banned successfully.', 'admin_success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/unban_user/<int:user_id>', methods=['POST'])
def unban_user(user_id):
    """Unban a user."""
    if not session.get('admin_logged_in'):
        flash('Unauthorized access.', 'admin_error')
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor()
    cursor.execute('UPDATE users SET banned = 0 WHERE id = %s', (user_id,))
    mysql.connection.commit()
    cursor.close()

    flash('User unbanned successfully.', 'admin_success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/remove_post/<int:post_id>', methods=['POST'])
def remove_post(post_id):
    """Remove a post."""
    if not session.get('admin_logged_in'):
        flash('Unauthorized access.', 'admin_error')
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor()
    cursor.execute('DELETE FROM replies WHERE postID = %s', (post_id,))
    cursor.execute('DELETE FROM posts WHERE id = %s', (post_id,))
    mysql.connection.commit()
    cursor.close()

    flash('Post removed successfully.', 'admin_success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/remove_reply/<int:reply_id>', methods=['POST'])
def remove_reply(reply_id):
    """Remove a reply."""
    if not session.get('admin_logged_in'):
        flash('Unauthorized access.', 'admin_error')
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor()
    cursor.execute('DELETE FROM replies WHERE id = %s', (reply_id,))
    mysql.connection.commit()
    cursor.close()

    flash('Reply removed successfully.', 'admin_success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/logout')
def admin_logout():
    """Admin logout."""
    session.clear()  # Clear all session data
    flash('You have been logged out as admin.', 'admin_success')
    return redirect(url_for('login'))

# Function to run scheduled tasks
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

# Function to start training in a separate thread
def start_training():
    try:
        print("Training started...")
        train_model()  # Call the training function from model.py
        print("Training completed.")
    except Exception as e:
        print(f"Error during training: {e}")

# Schedule training at a specific time (e.g., every day at 2 AM)
schedule.every().day.at("02:00").do(lambda: threading.Thread(target=start_training).start())

# Start the scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_schedule, daemon=True)
scheduler_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)