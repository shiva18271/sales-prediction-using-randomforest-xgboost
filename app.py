import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
from werkzeug.utils import secure_filename
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
import glob

import google.generativeai as genai 

app = Flask(__name__)
app.secret_key = '1234'

genai.configure(api_key="AIzaSyD5oIp0iZ_P5e8ZHZhndqJF5SnjzqsjZoQ")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

data = pd.read_csv('Walmart.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Week'] = data['Date'].dt.isocalendar().week
data.drop('Date', axis=1, inplace=True)

X = data.drop('Weekly_Sales', axis=1)
y = data['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

historical_avg_sales = data.groupby("Store")["Weekly_Sales"].mean().to_dict()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}
for model in models.values():
    model.fit(X_train_scaled, y_train)


@app.route("/reset", methods=["POST"])
def reset_uploads():
    upload_folder = app.config['UPLOAD_FOLDER']
    files = glob.glob(os.path.join(upload_folder, "*.csv"))
    
    if not files:
        
        return jsonify({"message": "No files to delete!", "redirect": url_for('home')})  # ✅ Return JSON response

    for f in files:
        os.remove(f)
    
    # flash("All uploaded CSV files have been deleted successfully!", "success")
    return jsonify({"message": "All uploaded CSV files have been deleted successfully!", "redirect": url_for('home')})  # ✅ Return JSON response


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(request.url)
        if not file.filename.endswith('.csv'):
            flash('Invalid file format! Only CSV allowed.', 'danger')
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            uploaded_data = pd.read_csv(filepath)
            required_columns = ['Store', 'Date', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
            if not all(col in uploaded_data.columns for col in required_columns):
                flash('Invalid file structure! Ensure correct columns.', 'danger')
                os.remove(filepath)
                return redirect(request.url)
        except Exception:
            flash('Error reading file!', 'danger')
            return redirect(request.url)
        
        return redirect(url_for('input_data', filename=filename))
    
    return render_template('upload.html')


@app.route('/input/<filename>')
def input_data(filename):
    return render_template('input.html', filename=filename)

@app.route('/results/<filename>')
def results(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        flash("The selected file has been deleted. Please upload a new file.", "danger")
        return redirect(url_for('upload_file'))

    data = pd.read_csv(filepath)

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['Week'] = data['Date'].dt.isocalendar().week

  
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

    required_columns = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Day', 'Week']
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        flash(f"Missing columns: {', '.join(missing_columns)}", 'danger')
        return redirect(url_for('upload_file'))

    numeric_columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].fillna(data[col].mean())

    scaled_data = scaler.transform(data[required_columns])
    
    predictions = {
        'Random Forest': models['Random Forest'].predict(scaled_data),
        'XGBoost': models['XGBoost'].predict(scaled_data),
        'Linear Regression': models['Linear Regression'].predict(scaled_data)
    }

    for model, pred in predictions.items():
        data[f"{model}_Prediction"] = pred

    metrics = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        metrics[model_name] = {
            'R2': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
        }

    graphs = {}
    
    if 'Store' in data.columns:
        store_preds = data.groupby('Store')[['Random Forest_Prediction', 'XGBoost_Prediction', 'Linear Regression_Prediction']].mean().reset_index()
        fig_store = px.bar(store_preds, 
                          x='Store', 
                          y=['Random Forest_Prediction', 'XGBoost_Prediction', 'Linear Regression_Prediction'],
                          barmode='group',
                          title='Average Predicted Sales by Store and Model',
                          labels={'value': 'Predicted Sales', 'variable': 'Model'})
        graphs['store_comparison'] = fig_store.to_html(full_html=False)
    
    model_comparison = pd.DataFrame({
        'Model': list(metrics.keys()),
        'R2 Score': [m['R2'] for m in metrics.values()],
        'RMSE': [m['RMSE'] for m in metrics.values()],
        'MAE': [m['MAE'] for m in metrics.values()]
    })
    fig_models = px.bar(model_comparison, 
                       x='Model', 
                       y=['R2 Score', 'RMSE', 'MAE'],
                       barmode='group',
                       title='Model Performance Comparison',
                       labels={'value': 'Score', 'variable': 'Metric'},
                       color_discrete_sequence=px.colors.qualitative.Pastel)
    graphs['model_comparison'] = fig_models.to_html(full_html=False)

    return render_template('results.html', 
                            data=data.to_dict(orient='records'), 
                            metrics=metrics,
                            graphs=graphs)



@app.route("/recommendations")
def recommendations():
    upload_folder = "uploads"
    uploaded_files = [f for f in os.listdir(upload_folder) if f.endswith(".csv")]

    if not uploaded_files:
        flash("No uploaded file found. Please upload a dataset first.", "warning")
        return redirect(url_for("upload_file"))

    latest_file = max(uploaded_files, key=lambda f: os.path.getctime(os.path.join(upload_folder, f)))
    filepath = os.path.join(upload_folder, latest_file)

    df = pd.read_csv(filepath)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["Week"] = df["Date"].dt.isocalendar().week
        df.drop("Date", axis=1, inplace=True)

    required_columns = ["Store", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Year", "Month", "Day", "Week"]
    
    if not all(col in df.columns for col in required_columns):
            flash("Missing required columns in the uploaded file!", "danger")
            return redirect(url_for("upload_file"))

    numeric_columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df[col] = df[col].fillna(df[col].mean())


    scaled_data = scaler.transform(df[required_columns])

    df["Predicted_Sales"] = models["Random Forest"].predict(scaled_data)

    stores_data = []
    for store in df["Store"].unique():
        store_data = df[df["Store"] == store]
        avg_predicted_sales = store_data["Predicted_Sales"].mean()
        avg_historical_sales = historical_avg_sales.get(store, 0)

        insights = {
            "Store": store,
            "Predicted_Avg_Sales": avg_predicted_sales,
            "Historical_Avg_Sales": avg_historical_sales,
            "Holiday_Flag": int(store_data["Holiday_Flag"].any()),
            "Avg_Temperature": store_data["Temperature"].mean(),
            "Avg_Unemployment": store_data["Unemployment"].mean()
        }
        stores_data.append(insights)

    prompt = f"""
    Based on the following sales prediction data, generate store-specific recommendations:

    {stores_data}

    Provide insights such as inventory management, promotional campaigns, and external factor impacts also check if holiady flag is there if there
    the indicate for holiday prep also . 
    Keep the suggestions professional and actionable.and give for each store seperately and for each store give only
    3-4 lines only 
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest") 
        response = model.generate_content(prompt)
        ai_recommendations = response.text
        return render_template("recommendations.html", recommendations=ai_recommendations.split("\n"))
    except Exception as e:
        flash(f"Error generating recommendations: {e}", "danger")
        return redirect(url_for("upload_file"))


@app.route('/comparison', methods=['GET', 'POST'])
def comparison():
    data = pd.read_csv('Walmart.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    data['Year'] = data['Date'].dt.year
    stores = sorted(data['Store'].unique())
    selected_stores = request.form.getlist('stores')

    if selected_stores:
        selected_stores = list(map(int, selected_stores))
        filtered_data = data[data['Store'].isin(selected_stores)]
        
        yearly_sales = filtered_data.groupby(['Store', 'Year'])['Weekly_Sales'].sum().reset_index()
        # avg_sales = filtered_data.groupby(['Store', 'Year'])['Weekly_Sales'].mean().reset_index()  # Now by Year
        avg_sales = filtered_data.groupby('Store')['Weekly_Sales'].mean().reset_index()
        holiday_sales = filtered_data.groupby(['Store', 'Year', 'Holiday_Flag'])['Weekly_Sales'].sum().unstack(fill_value=0).reset_index()

       
        holiday_sales.columns = ['Store', 'Year', 'Non-Holiday Sales', 'Holiday Sales']

        fuel_impact = filtered_data.groupby(['Store', 'Year'])['Fuel_Price'].mean().reset_index()
        cpi_impact = filtered_data.groupby(['Store', 'Year'])['CPI'].mean().reset_index()
        unemployment_impact = filtered_data.groupby(['Store', 'Year'])['Unemployment'].mean().reset_index()

        best_store = filtered_data.groupby('Store')['Weekly_Sales'].mean().idxmax()

        final_data = (
            yearly_sales
            # .merge(avg_sales, on=['Store', 'Year'], suffixes=('_Total', '_Avg'))
            .merge(avg_sales, on='Store', suffixes=('_Total', '_Avg'))
            .merge(holiday_sales, on=['Store', 'Year'])
            .merge(fuel_impact, on=['Store', 'Year'])
            .merge(cpi_impact, on=['Store', 'Year'])
            .merge(unemployment_impact, on=['Store', 'Year'])
        )

        final_data.rename(columns={
            'Weekly_Sales_Total': 'Total Sales',
            'Weekly_Sales_Avg': 'Average Sales',
            'Fuel_Price': 'Avg Fuel Price',
            'CPI': 'Avg CPI',
            'Unemployment': 'Avg Unemployment'
        }, inplace=True)

        final_data.fillna(0, inplace=True)


        final_data_list = final_data.to_dict(orient='records')
        final_data.fillna(0, inplace=True)  
        final_data_list = final_data.to_dict(orient='records')


        fig_yearly = px.bar(final_data, x='Year', y='Total Sales', color='Store', barmode='group', title='Yearly Sales Comparison')
        fig_avg = px.bar(final_data, x='Store', y='Average Sales', title='Average Sales per Store')
        fig_holiday = px.bar(final_data, x='Store', y=['Holiday Sales', 'Non-Holiday Sales'], barmode='group', title='Sales on Holidays vs Non-Holidays')
        fig_fuel = px.scatter(final_data, x='Avg Fuel Price', y='Total Sales', color='Store', title='Fuel Price Impact on Sales')
        fig_cpi = px.scatter(final_data, x='Avg CPI', y='Total Sales', color='Store', title='CPI Impact on Sales')
        fig_unemployment = px.scatter(final_data, x='Avg Unemployment', y='Total Sales', color='Store', title='Unemployment Impact on Sales')
        
        graphs = {
            'yearly_sales': fig_yearly.to_html(full_html=False),
            'avg_sales': fig_avg.to_html(full_html=False),
            'holiday_sales': fig_holiday.to_html(full_html=False),
            'fuel_impact': fig_fuel.to_html(full_html=False),
            'cpi_impact': fig_cpi.to_html(full_html=False),
            'unemployment_impact': fig_unemployment.to_html(full_html=False),
        }

        return render_template('comparison.html', stores=stores, selected_stores=selected_stores, graphs=graphs, best_store=best_store, final_data=final_data_list)

    return render_template('comparison.html', stores=stores, selected_stores=[], graphs={}, best_store=None, final_data=[])

@app.route('/dashboard')
def dashboard():
    data = pd.read_csv('Walmart.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month

    sales_trends = data.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
    store_sales = data.groupby('Store')['Weekly_Sales'].sum().reset_index()
    holiday_sales = data.groupby('Holiday_Flag')['Weekly_Sales'].mean().reset_index()
    temp_sales = data.groupby('Temperature')['Weekly_Sales'].mean().reset_index()
    fuel_sales = data.groupby('Fuel_Price')['Weekly_Sales'].mean().reset_index()
    cpi_sales = data.groupby('CPI')['Weekly_Sales'].mean().reset_index()
    unemployment_sales = data.groupby('Unemployment')['Weekly_Sales'].mean().reset_index()

    fig_sales = px.line(sales_trends, x='Month', y='Weekly_Sales', color='Year', 
                        title='📈 Monthly Sales Trends', markers=True)
    fig_sales.write_html('static/sales_trends.html')

    fig_store = px.bar(store_sales, x='Store', y='Weekly_Sales', color='Store',
                       text_auto='.2s', title='🏪 Store Sales Performance')
    fig_store.write_html('static/store_sales.html')

    fig_holiday = px.pie(holiday_sales, names='Holiday_Flag', values='Weekly_Sales',
                         title='🎉 Holiday vs. Non-Holiday Sales', hole=0.3)
    fig_holiday.write_html('static/holiday_sales.html')

    fig_temp = px.scatter(temp_sales, x='Temperature', y='Weekly_Sales', 
                          title='🌡️ Temperature vs Sales', color='Weekly_Sales', 
                          size='Weekly_Sales', size_max=10)
    fig_temp.write_html('static/temperature_sales.html')


    fig_fuel = px.bar(
    fuel_sales, 
    x='Fuel_Price', 
    y='Weekly_Sales', 
    color='Weekly_Sales', 
    color_continuous_scale="Blues",
    title='⛽ Fuel Price Impact on Sales'
)

    fig_fuel.update_layout(
    plot_bgcolor='white', 
    title_font=dict(size=20, family="Arial", color="black"), 
    xaxis=dict(showgrid=False),  
    yaxis=dict(showgrid=False),
    coloraxis_colorbar=dict(title="Sales Volume") 
)   

    fig_fuel.write_html('static/fuel_sales.html')


    fig_cpi = px.scatter(cpi_sales, x='CPI', y='Weekly_Sales', 
                      color='Weekly_Sales', 
                      trendline="ols", 
                      title='📉 CPI vs Sales (Impact of Inflation)',
                      labels={'CPI': 'Consumer Price Index', 'Weekly_Sales': 'Weekly Sales'},
                      color_continuous_scale='blues') 

    fig_cpi.write_html('static/cpi_sales.html')



    return render_template('dashboard.html',
                           sales_trends_url='static/sales_trends.html',
                           store_sales_url='static/store_sales.html',
                           holiday_sales_url='static/holiday_sales.html',
                           temp_sales_url='static/temperature_sales.html',
                           fuel_sales_url='static/fuel_sales.html',
                           cpi_sales_url='static/cpi_sales.html',
                           total_sales=f"{data['Weekly_Sales'].sum():,.2f}",
                           avg_sales=f"{data['Weekly_Sales'].mean():,.2f}",
                           avg_unemployment=f"{data['Unemployment'].mean():,.2f}")



if __name__ == '__main__':
    app.run(debug=True)
