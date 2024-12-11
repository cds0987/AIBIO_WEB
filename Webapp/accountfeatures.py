from flask import Blueprint, render_template, request, current_app,session
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ResearchFeatures.Predict import prediction
from functools import wraps

def userlogin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return render_template('Authentication/login.html')  # Redirect to login if not logged in
        return f(*args, **kwargs)
    return decorated_function



acfeature_bp = Blueprint('acfeatures', __name__)
IMG_PAY = "static/pay"
os.makedirs(IMG_PAY, exist_ok=True)

import json
# Load or create the tracking file
def load_usage_data(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}

def save_usage_data(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)



payment_track_file='UserTracking/payment_tracking.json'
#/proccessplus

from werkzeug.utils import secure_filename
from PIL import Image
from ResearchFeatures.SmilesImages import Images_Smiles
import urllib.parse
images = Images_Smiles()
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@acfeature_bp.route('/proccessplus', methods=['POST', 'GET'])
@userlogin_required
def processplus():
    # Retrieve user session details
    user_id = session.get('user_id')
    user_email = session.get('email')
    update = session.get('update', 'Plus')  # Default to "Plus" if not set

    # Validate and save the uploaded file
    file = request.files.get('paymentimage')
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{user_id}_{user_email}_{update}.png")
        image_path = os.path.join(IMG_PAY, filename)
        file.save(image_path)

        # Normalize path for use in the HTML
        proof_image_url = image_path.replace("\\", "/")
        print('proof_image_url')
        # Define payment details
        process = 'Pending'
        payment_entry = {
            'email': user_email,
            'plan': update,
            'status': process,
            'proof_image': proof_image_url
        }

        # Save payment data
        payment_data = load_usage_data(payment_track_file)
        payment_data[user_id] = payment_entry
        save_usage_data(payment_data, payment_track_file)

        # Message and proof image
        message = (
            "Your payment is being processed. Please allow up to 24 hours for admin approval."
        )
        return render_template('AccountFeatures/processupdate.html', message=message, proof_image=proof_image_url)

    else:
        return render_template(
            'AccountFeatures/processupdate.html',
            message="Invalid file upload. Please upload a valid image. We only allow jpg,png,jpeg"
        )
    
from utils.database_utils import DatabaseManager
db_manager=DatabaseManager()
@acfeature_bp.route('/movechangepassword', methods=['GET', 'POST'])
def change_password():
    if request.method == 'POST':
        userID=request.form['user_id']
        current_password = request.form['current-password']
        new_password = request.form['new-password']
        confirm_password = request.form['confirm-password']
        user=db_manager.get_user_by_id(userID)
        if user is None:
            return render_template('AccountFeatures/change-password.html', error='User not found.')
        # Check if the current password is correct
        if not current_password or current_password != user.Password:
            return render_template('change-password.html', error='Current password is incorrect.')
        db_manager.change_password(userID, new_password)
        return render_template('AccountFeatures/change-password.html', success='Password changed successfully.')
    
@acfeature_bp.route('/changeemail', methods=['GET', 'POST'])
def changeemail():
    if request.method == 'POST':
        userID=request.form['user_id']
        new_email = request.form['new-email']
        user=db_manager.get_user_by_id(userID)
        if user is None:
            return render_template('AccountFeatures/change-email.html', error='User not found.')
        db_manager.change_email(userID, new_email)
        return render_template('AccountFeatures/change-email.html', success='Email changed successfully.')
    