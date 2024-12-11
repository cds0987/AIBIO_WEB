from flask import Blueprint, render_template

general_bp = Blueprint('general', __name__)

@general_bp.route('/')
def home():
    return render_template('index.html')

@general_bp.route('/about')
def about():
    return render_template('Introduce/about.html')

@general_bp.route('/media')
def media():
    return render_template('Introduce/media.html')


@general_bp.route('/references')
def references():
    return render_template('Introduce/references.html')

@general_bp.route('/modelcapacity')
def model_capacity():
    return render_template('Introduce/model.html')  # Make sure you have this template