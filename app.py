from flask import Flask
from Webapp.auth_views import auth_bp
from Webapp.researchfeatures import feature_bp
from Webapp.general_views import general_bp
from Webapp.mainpage import mainpage_bth
from Webapp.accountfeatures import acfeature_bp
from Webapp.adminfeatures import adminfeatures_bth
app = Flask(__name__)
app.config['SECRET_KEY'] = '1537'
app.config.from_pyfile('config.py')
# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(feature_bp, url_prefix='/features')
app.register_blueprint(general_bp)
app.register_blueprint(mainpage_bth, url_prefix='/mainpage')
app.register_blueprint(acfeature_bp, url_prefix='/acfeatures')
app.register_blueprint(adminfeatures_bth, url_prefix='/adfeatures')
# Database manager instance



