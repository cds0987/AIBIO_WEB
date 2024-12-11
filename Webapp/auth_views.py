from flask import Blueprint, render_template, request, redirect, url_for, session
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database_utils import DatabaseManager
auth_bp = Blueprint('auth', __name__)

@auth_bp.route("/signup", methods=["GET", "POST"])
def register():
    db_manager=DatabaseManager()
    if request.method == "POST":
        user_id = request.form.get("ID")
        full_name = request.form.get("FullName")
        email = request.form.get("Email")
        password = request.form.get("Password")
        confirm_password = request.form.get("ConfirmPassword")

        if password != confirm_password:
            return render_template("Authentication/register.html", status="error", message="Passwords do not match.")
        try:
            db_manager.add_user(user_id, full_name, email, password, level=1)
            session["user_id"] = user_id
            return redirect(url_for("auth.login"))
        except Exception as e:
            return render_template("Authentication/register.html", status="error", message=str(e))
    return render_template("Authentication/register.html")



@auth_bp.route("/")
def ul():
    # Render the template for the user main page
    return render_template("Authentication/login.html")
@auth_bp.route("/adminlog")
def al():
    # Render the template for the user main page
    return render_template("Authentication/adminlogin.html")


@auth_bp.route("/userformlogin", methods=["GET", "POST"])
def userlogin():
    db_manager=DatabaseManager()
    if request.method == "POST":
        user_id = request.form.get("user_id")
        password = request.form.get("password")
        user = db_manager.get_user_by_id(user_id)
        if user and user.Password == password:
            session["user_id"] = user.ID
            session["full_name"] = user.FullName
            session["email"] = user.Email
            session["level"] = user.Level
            return redirect(url_for("mainpage.usermainpage"))
        else:
            return render_template("Authentication/login.html", status="error", message="Invalid credentials.")
    return render_template("Authentication/login.html")



@auth_bp.route("/adminformlogin", methods=["GET", "POST"])
def adminlogin():
    db_manager=DatabaseManager()
    if request.method == "POST":
        admin_id = request.form.get("admin_id")
        password = request.form.get("password")
        user = db_manager.get_admin(admin_id)
        if user and user.Password == password:
            session["admin_id"] = user.ID
            session["full_name"] = user.FullName
            session["email"] = user.Email
            return redirect(url_for("mainpage.adminpage"))
        else:
            return render_template("Authentication/adminlogin.html", status="error", message="Invalid credentials.")
    return render_template("Authentication/adminlogin.html")


@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("general.home"))
