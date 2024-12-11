from flask import Blueprint, render_template, request, current_app,session
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ResearchFeatures.Predict import prediction
from functools import wraps

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            return render_template('Authentication/adminlogin.html')  # Redirect to login if not logged in
        return f(*args, **kwargs)
    return decorated_function

# Blueprint for the main page
adminfeatures_bth = Blueprint('adminfeatures', __name__)





import json
from datetime import datetime
from utils.database_utils import DatabaseManager
db_manager=DatabaseManager()
payment_track_file='UserTracking/payment_tracking.json'
@adminfeatures_bth.route('/update_user', methods=['GET', 'POST'])
@admin_required
def update_user():
    if request.method == 'POST':
        user_id = request.form['user_id']
        new_level = request.form['new_level']

        # Load payment tracking data from the JSON file
        with open(payment_track_file, 'r') as file:
            payment_data = json.load(file)

        # Check if the user exists in the JSON data
        if user_id not in payment_data:
            users = [
                {
                    "user_id": uid,
                    **details
                } 
                for uid, details in payment_data.items()
            ]
            return render_template(
                'updateuser.html', 
                message='User not found. Please check the User ID.', 
                users=users
            )

        # Update user level and status in the JSON data
        payment_data[user_id]['level'] = new_level
        payment_data[user_id]['status'] = 'completed'
        payment_data[user_id]['completion_day'] = datetime.now().strftime('%Y-%m-%d')
        db_manager.update_user_level(user_id, new_level)

        # Save the updated data back to the JSON file
        with open(payment_track_file, 'w') as file:
            json.dump(payment_data, file, indent=4)

        # Reload the updated user list
        users = [
            {
                "user_id": uid,
                **details
            } 
            for uid, details in payment_data.items()
        ]

        return render_template(
            'AdminFeatures/updateuser.html', 
            message='User level, status, and completion day updated successfully.', 
            users=users
        )
    else:
        # Handle GET requests by loading all users
        with open(payment_track_file, 'r') as file:
            payment_data = json.load(file)
        
        users = [
            {
                "user_id": uid,
                **details
            } 
            for uid, details in payment_data.items()
        ]

        return render_template('AdminFeatures/updateuser.html', users=users)







import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
# Ensure the directory for plots exists
PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def create_payment_plots(df, plot_dir="PLOT_DIR"):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Bar Plot: Top 5 Users by Payment
    bar_path = os.path.join(plot_dir, "payment_bar.png")
    plt.figure(figsize=(8, 6))
    top_5_df = df.sort_values(by="payment", ascending=False).head(5)
    top_5_df.plot(kind="bar", x="name", y="payment", color="skyblue", legend=False)
    plt.title("Top 5 Users by Payment")
    plt.xlabel("Name")
    plt.ylabel("Payment")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()

    # Pie Chart (Payment Ranges)
    bins = [0, 450, 600, 800, float('inf')]
    labels = ['<450', '451-599', '600-799', '>=800']
    df['payment_range'] = pd.cut(df['payment'], bins=bins, labels=labels, right=False)
    range_counts = df['payment_range'].value_counts(sort=False)
    pie_path = os.path.join(plot_dir, "payment_pie.png")
    plt.figure(figsize=(8, 6))
    plt.pie(
        range_counts.values, 
        labels=range_counts.index, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=plt.cm.Set3.colors
    )
    plt.title("Payment Distribution by Ranges")
    plt.tight_layout()
    plt.savefig(pie_path)
    plt.close()

    # Histogram: Payment Distribution
    hist_path = os.path.join(plot_dir, "payment_hist.png")
    plt.figure(figsize=(8, 6))
    plt.hist(df['payment'], bins=10, color='teal', edgecolor='black', alpha=0.7)
    plt.title("Payment Distribution")
    plt.xlabel("Payment")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    # Box Plot: Payment Spread
    box_path = os.path.join(plot_dir, "payment_box.png")
    plt.figure(figsize=(8, 6))
    plt.boxplot(df['payment'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title("Payment Spread")
    plt.xlabel("Payment")
    plt.tight_layout()
    plt.savefig(box_path)
    plt.close()

    # Cumulative Number of Users and Revenue Plot (Line Curve)
    user_revenue_path = os.path.join(plot_dir, "user_revenue.png")
    plt.figure(figsize=(8, 6))
    df_sorted = df.sort_values(by="payment")
    df_sorted['cumulative_revenue'] = df_sorted['payment'].cumsum()
    users = range(1, len(df_sorted) + 1)
    cumulative_revenue = df_sorted['cumulative_revenue'].values
    plt.plot(users, cumulative_revenue, marker='o', linestyle='-', color='blue', markersize=5)
    plt.scatter(users, cumulative_revenue, color='red', zorder=5)
    plt.title("Cumulative Revenue by Number of Users")
    plt.xlabel("Number of Users")
    plt.ylabel("Cumulative Revenue")
    plt.tight_layout()
    plt.savefig(user_revenue_path)
    plt.close()

    # Scatter Plot: Payment vs. User Count
    scatter_path = os.path.join(plot_dir, "payment_scatter.png")
    plt.figure(figsize=(8, 6))
    plt.scatter(df.index, df['payment'], color='purple', alpha=0.6)
    plt.title("Payment vs. User Count")
    plt.xlabel("User Index")
    plt.ylabel("Payment")
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()

    # Linear Regression: Payment vs. Users
    regression_path = os.path.join(plot_dir, "payment_regression.png")
    plt.figure(figsize=(8, 6))
    df['user_index'] = df['name'].factorize()[0]
    X = df['user_index'].values.reshape(-1, 1)
    y = df['payment'].values
    model = LinearRegression().fit(X, y)
    predictions = model.predict(X)
    plt.scatter(df['user_index'], df['payment'], color='purple', alpha=0.6, label='Data points')
    plt.plot(df['user_index'], predictions, color='red', linewidth=2, label='Regression Line')
    plt.title("Payment vs. Users with Linear Regression")
    plt.xlabel("Users")
    plt.ylabel("Payment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(regression_path)
    plt.close()

    # Violin Plot: Payment Distribution by Group
    violin_path = os.path.join(plot_dir, "payment_violin.png")
    plt.figure(figsize=(8, 6))
    if 'category' in df.columns:
        sns.violinplot(x='category', y='payment', data=df, inner="quart")
    else:
        sns.violinplot(x='payment_range', y='payment', data=df, inner="quart")
    plt.title("Payment Distribution by Group")
    plt.tight_layout()
    plt.savefig(violin_path)
    plt.close()


    # New Diagram 2: Density Plot (Payment Distribution)
    density_path = os.path.join(plot_dir, "payment_density.png")
    plt.figure(figsize=(8, 6))
    sns.kdeplot(df['payment'], shade=True, color="green")
    plt.title("Payment Density Plot")
    plt.xlabel("Payment")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(density_path)
    plt.close()

    return {
        "bar": "/static/plots/payment_bar.png",
        "pie": "/static/plots/payment_pie.png",
        "hist": "/static/plots/payment_hist.png",
        "box": "/static/plots/payment_box.png",
        "user_revenue": "/static/plots/user_revenue.png",
        "scatter": "/static/plots/payment_scatter.png",
        "regression": "/static/plots/payment_regression.png",
        "violin": "/static/plots/payment_violin.png",
        "density": "/static/plots/payment_density.png"
    }






@adminfeatures_bth.route('/analyzedata', methods=['GET', 'POST'])
@admin_required
def analyzedata():
    # Mock database manager call
    users = db_manager.get_all_users()

    # Extract names and payments
    data = {
        'name': [user["FullName"] for user in users],
        'payment': [user["Payment"] for user in users]
    }

    df = pd.DataFrame(data)

    # Create all plots
    plot_urls = create_payment_plots(df, PLOT_DIR)

    # Compute notable indices
    mean = df['payment'].mean()
    median = df['payment'].median()
    mode = df['payment'].mode().tolist()  # Handle multiple modes
    std_dev = df['payment'].std()
    variance = df['payment'].var()
    min_payment = df['payment'].min()
    max_payment = df['payment'].max()
    payment_range = max_payment - min_payment
    quartiles = df['payment'].quantile([0.25, 0.5, 0.75]).to_dict()
    iqr = quartiles[0.75] - quartiles[0.25]
    skewness = df['payment'].skew()
    kurtosis = df['payment'].kurt()
    total_users = len(df)  # Number of unique users
    total_revenue = df['payment'].sum()
    # Pass data to template
    return render_template(
        'AdminFeatures/analyzedata.html',
        plot_urls=plot_urls,
        mean=mean,
        median=median,
        mode=mode,
        std_dev=std_dev,
        variance=variance,
        min_payment=min_payment,
        max_payment=max_payment,
        payment_range=payment_range,
        quartiles=quartiles,
        iqr=iqr,
        skewness=skewness,
        kurtosis=kurtosis,
        total_users= total_users,
        total_revenue= total_revenue
    )