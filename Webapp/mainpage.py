from flask import Blueprint, render_template
from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, session

def userlogin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return render_template('Authentication/login.html')  # Redirect to login if not logged in
        return f(*args, **kwargs)
    return decorated_function
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            return render_template('Authentication/adminlogin.html')  # Redirect to login if not logged in
        return f(*args, **kwargs)
    return decorated_function

# Blueprint for the main page
mainpage_bth = Blueprint('mainpage', __name__)



# Route for the user main page
@mainpage_bth.route("/")
@userlogin_required
def usermainpage():
    # Render the template for the user main page
    return render_template("MainPage/userpage.html")


@mainpage_bth.route("/admin")
@admin_required
def adminpage():
    # Render the template for the user main page
    return render_template("MainPage/adminpage.html")




@mainpage_bth.route("/movepredict")
@userlogin_required
def movepredict():
    return render_template('ResearchFeatures/predict.html')


@mainpage_bth.route("/movisualize")
@userlogin_required
def movisualize():
    return render_template('ResearchFeatures/visualize.html')


@mainpage_bth.route("/movecaption")
@userlogin_required
def movecaption():
    current_level=session.get('level')
    if current_level < 2:
        #send notify warning not allow to access
        return render_template('MainPage/userpage.html', error="You don't have permission to access this page.")
    return render_template('ResearchFeatures/describe.html')




@mainpage_bth.route("/movegenerate")
@userlogin_required
def movegenerate():
    current_level=session.get('level')
    if current_level < 2:
        #send notify warning not allow to access
        return render_template('MainPage/userpage.html', error="You don't have permission to access this page.")
    return render_template('ResearchFeatures/generate.html')




@mainpage_bth.route("/movemoleculeimage")
@userlogin_required
def movemoleculeimage():
    current_level=session.get('level')
    if current_level < 2:
        #send notify warning not allow to access
        return render_template('MainPage/userpage.html', error="You don't have permission to access this page.")
    return render_template('ResearchFeatures/moleculeimage.html')






@mainpage_bth.route("/movesearchsmiles")
@userlogin_required
def movesearchsmiles():
    current_level=session.get('level')
    if current_level < 2:
        #send notify warning not allow to access
        return render_template('MainPage/userpage.html', error="You don't have permission to access this page.")
    return render_template('ResearchFeatures/searchsmiles.html')


@mainpage_bth.route('/moveconditionalsearch', methods=['GET', 'POST'])
@userlogin_required
def movesearchconstraints():
    current_level=session.get('level')
    if current_level < 3:
        #send notify warning not allow to access
        return render_template('MainPage/userpage.html', error="You don't have permission to access this page.")
    return render_template('ResearchFeatures/conditionalsearch.html')



@mainpage_bth.route('/moveretrive', methods=['GET', 'POST'])
@userlogin_required
def moveretrive():
    current_level=session.get('level')
    if current_level < 3:
        #send notify warning not allow to access
        return render_template('MainPage/userpage.html', error="You don't have permission to access this page.")
    return render_template('ResearchFeatures/retrivedocuments.html')


@mainpage_bth.route('/moveadvancedtool', methods=['GET', 'POST'])
@userlogin_required
def moveadvancedtool():
    current_level=session.get('level')
    if current_level < 3:
        #send notify warning not allow to access
        return render_template('MainPage/userpage.html', error="You don't have permission to access this page.")
    return render_template('ResearchFeatures/advancedtool.html')


@mainpage_bth.route('/feedback', methods=['GET', 'POST'])
def feedback():
    return render_template('ResearchFeatures/feedback.html')


@mainpage_bth.route('/demo')
def demo():
    return render_template('ResearchFeatures/demo.html')

@mainpage_bth.route('/upgrade-account', methods=['GET', 'POST'])
def upgrade_account():
     return render_template('AccountFeatures/upgrade-account.html')


@mainpage_bth.route('/processplusproof', methods=['POST', 'GET'])
@userlogin_required
def processplusproof():
    session['update'] ='2' 
    return render_template('AccountFeatures/processupdate.html')



@mainpage_bth.route('/processproproof', methods=['POST', 'GET'])
@userlogin_required
def processproproof():
    session['update'] ='3'
    return render_template('AccountFeatures/processupdate.html')

from utils.database_utils import DatabaseManager
db_manager=DatabaseManager()
@mainpage_bth.route("/mouser", methods=['GET', 'POST'])
@admin_required
def moveusers():
    #get allusers
    users=db_manager.get_all_users()
    return render_template('AdminFeatures/usersmanagement.html',users=users)


import json
import os
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
@mainpage_bth.route('/updateuser', methods=['GET', 'POST'])
@admin_required
def updateuser():
    # Load data from payment_tracking.json
    payment_data = load_usage_data(payment_track_file)
    
    # Convert the data into a list of dictionaries for easier rendering
    users = [
        {
            "user_id": user_id,
            **details
        } 
        for user_id, details in payment_data.items()
    ]

    return render_template('AdminFeatures/updateuser.html', users=users)



@mainpage_bth.route('/movetrack')
@admin_required
def movetrack():
    results=db_manager.getalltracking()
    return render_template('AdminFeatures/trackusers.html',results=results)


@mainpage_bth.route('/moverecord', methods=['GET', 'POST'])
@admin_required
def moverecord():
    results=db_manager.get_all_records()
    new_reults=[]
    for result in results:
        time=result['Time']
        if isinstance(time, str) and 'hour' in time.lower():
            time=time.replace('hour', '').strip()
            result['Time'] = time
        new_reults.append(result)
    print(new_reults)
    return render_template('AdminFeatures/record.html',results=new_reults)


@mainpage_bth.route('/change-password', methods=['GET', 'POST'])
def movechangepassword():
    return render_template('AccountFeatures/change-password.html')


@mainpage_bth.route('/change-email', methods=['GET', 'POST'])
def movechangeemail():    
    return render_template('AccountFeatures/change-email.html')