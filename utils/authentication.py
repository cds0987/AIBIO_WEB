from flask import session, render_template
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return render_template('Authentication/login.html')  # Redirect to login if not logged in
        return f(*args, **kwargs)
    return decorated_function
