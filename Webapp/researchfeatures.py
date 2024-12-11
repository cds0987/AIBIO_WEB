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



feature_bp = Blueprint('features', __name__)





freeprediction = prediction()
@feature_bp.route('/predict', methods=['POST'])
@userlogin_required
def predict():
    smiles = request.form['smiles']
    selected_model = request.form['use_model']
    try:
        if selected_model == '1':
            results = freeprediction.predict_m1(smiles)
        elif selected_model == '2':
            results = freeprediction.predict_m2(smiles)
        else:
            raise ValueError("Invalid model selection")
        return render_template('ResearchFeatures/predict.html', prediction=results)
    except Exception as e:
        return render_template('ResearchFeatures/predict.html', error=str(e))

from ResearchFeatures.Visualize import Visualize
visualize=Visualize()
@feature_bp.route('/visualize', methods=['POST'])
@userlogin_required
def proccess():
    smiles = request.form['smiles']
    img_path = visualize.smileImage(smiles)
    print('img_path',img_path)
    return render_template('ResearchFeatures/visualize.html', smiles=smiles, img_path=img_path)

#static/temp_images/[Cl].CC(C)NCC(O)COc1cccc2ccccc12.png


from ResearchFeatures.CaptionSmile import CaptionSmiles
describer=CaptionSmiles()
@feature_bp.route('/describe', methods=['POST'])
@userlogin_required
def generate():
    smiles = request.form['smiles']
    text = describer.caption(smiles)
    return render_template('ResearchFeatures/describe.html', smiles=smiles, caption=text)



from ResearchFeatures.Text2Smiles import Generate
generator=Generate()
@feature_bp.route('/generate', methods=['POST'])
@userlogin_required
def generate_smiles():
    text = request.form['generate']
    smiles = generator.generate(text)
    img_path = visualize.smileImage(smiles)
    return render_template('ResearchFeatures/generate.html', smiles=smiles, img_path=img_path)






from werkzeug.utils import secure_filename
from PIL import Image
from ResearchFeatures.SmilesImages import Images_Smiles
import urllib.parse
images = Images_Smiles()
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@feature_bp.route('describe_image', methods=['GET', 'POST'])
@userlogin_required
def upload_file():
    if request.method == 'POST':
        file = request.files['moleculeimage']
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            upload_folder = current_app.config['UPLOAD_FOLDER']
            filepath = os.path.join(upload_folder, filename)
            
            # Ensure the upload directory exists
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            
            # Save the file to the upload folder
            file.save(filepath)
            
            # Process the image and generate SMILES
            image = Image.open(filepath)
            smiles = images.Image2Smiles(image)
            description = describer.caption(smiles)  # Example description
            
            # Generate the relative URL for the uploaded image
            img_url = f"/static/uploads/{filename}"
            
            return render_template(
                'ResearchFeatures/moleculeimage.html',
                smiles=smiles,
                img_path=img_url,
                description=description
            )
    
    return render_template('ResearchFeatures/moleculeimage.html')




from ResearchFeatures.SearchSmiles import search_engine
search_engine_instance = search_engine()
@feature_bp.route('/searchsmiles', methods=['POST'])
@userlogin_required
def search_smiles():
    smiles = request.form['smiles']  # Get the input SMILES from the form
    k = 5  # Number of similar SMILES to retrieve
    # Initialize the search engine and retrieve similarities
      # Ensure search_engine is properly initialized
    similarities = search_engine_instance.search(smiles, k)
    # Format the similarities into a list of dictionaries with 'category' and 'smiles'
    formatted_similarities = []
    for category, smiles_list in similarities.items():
        for smile in smiles_list:
            formatted_similarities.append({
                'category': category,
                'smiles': smile
            })
    # Render the template with the SMILES and similarities
    return render_template('ResearchFeatures/searchsmiles.html', smiles=smiles, similarities=formatted_similarities)


from ResearchFeatures.SearchConstraints import SearchConstraintsEngine
search_constraints_engine=SearchConstraintsEngine()
@feature_bp.route('/searchconstraints', methods=['POST'])
@userlogin_required

def search_constraints():
    property = {
            'bbbp': int(request.form['bbbp']),
            'bace': int(request.form['bace']),
            'fda': int(request.form['fda']),
            'ctox': int(request.form['ctox']),
            'hiv': int(request.form['hiv']),
        }
    k = int(request.form['k'])
    result=search_constraints_engine.search(property,k)
    return render_template('ResearchFeatures/conditionalsearch.html',smiles_list=result)




from AI_Model.SearchDocuments import RetrieveInformation
retriever=RetrieveInformation()
@feature_bp.route('/retrive', methods=['POST'])
@userlogin_required

def search_information():
    if request.method == 'POST':
        query = request.form['query']
        results = retriever.search_document(query)
        return render_template('ResearchFeatures/retrivedocuments.html', results=results)
    return render_template('ResearchFeatures/retrivedocuments.html', results=None)



from ResearchFeatures.AdvancedPredict import Adprediction
advancedprediction=Adprediction()
@feature_bp.route('/advanced', methods=['POST'])
@userlogin_required

def advancedproccess():
    smiles = request.form['smiles']
    bbbp,bace,ctox,fda,hiv,esol,freesolv,lipophilicity,qm7,E1CC2,E2CC2,E1PBE0,E2PBE0,E1CAM,E2CAM=advancedprediction.predict(smiles)
    img_path = visualize.smileImage(smiles)
    description=describer.caption(smiles)
    similarities=search_engine_instance.search(smiles,5)
    formatted_similarities = []
    for category, smiles_list in similarities.items():
        for smile in smiles_list:
            formatted_similarities.append({
                'category': category,
                'smiles': smile
            })
    results={
        'bbbp':bbbp,
        'bace':bace,
        'ctox':ctox,
        'fda':fda,
        'hiv':hiv,
        'esol':esol,
        'freesolv':freesolv,
        'lipophilicity':lipophilicity,
        'qm7':qm7,
        'e1cc2':E1CC2,
        'e2cc2':E2CC2,
        'e1pbe0':E1PBE0,
        'e2pbe0':E2PBE0,
        'e1cam':E1CAM,
        'e2cam':E2CAM
    }
    return render_template('ResearchFeatures/advancedtool.html',results=results,img_path=img_path, smiles=smiles,description=description, similarities=formatted_similarities)


from utils.database_utils import DatabaseManager

@feature_bp.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    db_manager=DatabaseManager()
    # Get form data
    user_id = request.form.get('user-id')
    if user_id!=session.get('user_id'):
        return render_template('ResearchFeatures/feedback.html', message="Invalid user ID Enter your ID.")
    model_name = request.form.get('model')
    feedback = request.form.get('feedback')
    try:
       db_manager.change_feedback(user_id, model_name, feedback)
       return render_template('ResearchFeatures/feedback.html', message="Thank you for your feedback!")
    except Exception as e:
        return render_template('ResearchFeatures/feedback.html', message=f"An error occurred: {str(e)}")
    

USAGE_TRACKING_FILE = 'UserTracking/usage_tracking.json'

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

@feature_bp.route('/demoprocess', methods=['POST'])
def demoprocess():
    # Load the current usage data
    usage_data = load_usage_data(USAGE_TRACKING_FILE)

    # Identify the user (e.g., by IP address)
    user_identifier = request.remote_addr

    # Get user usage data or initialize it
    user_info = usage_data.get(user_identifier, {'usage_count': 0, 'max_exceeded': False})

    # Check if the user has exceeded the limit
    if user_info['max_exceeded']:
        return render_template('ResearchFeatures/demo.html', error="You have exceeded the maximum allowed usage limit. Register to continue.")

    # Increment usage count
    user_info['usage_count'] += 1

    # Check if the usage count exceeds the limit
    if user_info['usage_count'] > 3:
        user_info['max_exceeded'] = True
        usage_data[user_identifier] = user_info
        save_usage_data(usage_data,USAGE_TRACKING_FILE)
        return render_template('ResearchFeatures/demo.html', error="You have exceeded the maximum allowed usage limit. Register to continue.")

    # Save updated usage data
    usage_data[user_identifier] = user_info
    save_usage_data(usage_data, USAGE_TRACKING_FILE)

    # Perform the prediction process
    smiles = request.form['smiles']
    selected_model = request.form['use_model']
    results = {}

    try:
        # Validate and generate molecular image
        img_path = visualize.smileImage(smiles)
        if not img_path:
            return render_template('ResearchFeatures/demo.html', error="Invalid SMILES string. Please check your input and try again.")

        # Perform model predictions
        if selected_model == '1':
            bbbp, bace, ctox, fda, esol, freesolv, lipophilicity, qm7 = freeprediction.predict_m1(smiles)
        elif selected_model == '2':
            bbbp, bace, ctox, fda, esol, freesolv, lipophilicity, qm7 = freeprediction.predict_m2(smiles)
        else:
            return render_template('ResearchFeatures/demo.html', error="Invalid model selection.")
        
        # Compile results
        results = {
            'bbbp': bbbp,
            'bace': bace,
            'ctox': ctox,
            'fda': fda,
            'esol': esol,
            'freesolv': freesolv,
            'lipophilicity': lipophilicity,
            'qm7': qm7
        }

        # Render template with results and image path
        return render_template('ResearchFeatures/demo.html', prediction=results, img_path=img_path)

    except Exception as e:
        return render_template('ResearchFeatures/demo.html', error=f"An error occurred: {str(e)}")
