# @app.route('/recommendations')
# def recommendations():
#     # Get the latest uploaded file
#     upload_folder = app.config['UPLOAD_FOLDER']
#     uploaded_files = [f for f in os.listdir(upload_folder) if f.endswith('.csv')]
    
#     if not uploaded_files:
#         flash("No uploaded file found. Please upload a dataset first.", "warning")
#         return redirect(url_for('upload_file'))

#     latest_file = max(uploaded_files, key=lambda f: os.path.getctime(os.path.join(upload_folder, f)))
#     filepath = os.path.join(upload_folder, latest_file)

#     # Load uploaded data
#     data = pd.read_csv(filepath)

#     # Convert Date column to datetime and extract features
#     if 'Date' in data.columns:
#         data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
#         data['Year'] = data['Date'].dt.year
#         data['Month'] = data['Date'].dt.month
#         data['Day'] = data['Date'].dt.day
#         data['Week'] = data['Date'].dt.isocalendar().week
#         data.drop('Date', axis=1, inplace=True)

#     # Define required columns
#     required_columns = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Day', 'Week']
#     if not all(col in data.columns for col in required_columns):
#         flash("Missing required columns in the uploaded file!", "danger")
#         return redirect(url_for('upload_file'))

#     # Handle missing or invalid data in numeric columns
#     numeric_columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
#     for col in numeric_columns:
#         # Convert column to numeric, setting errors='coerce' to turn invalid values into NaN
#         data[col] = pd.to_numeric(data[col], errors='coerce')
        
#         # Replace NaN values with the column mean
#         data[col] = data[col].fillna(data[col].mean())

#     # Scale the data
#     scaled_data = scaler.transform(data[required_columns])

#     # Generate predictions
#     data['RF_Prediction'] = models['Random Forest'].predict(scaled_data)

#     # Load historical sales data (Assuming Walmart.csv contains historical sales)
#     historical_data = pd.read_csv('Walmart.csv')
#     historical_avg_sales = historical_data.groupby('Store')['Weekly_Sales'].mean().to_dict()

#     recommendations = []

#     for store in data['Store'].unique():
#         store_predictions = data[data['Store'] == store]['RF_Prediction']
#         predicted_avg = store_predictions.mean()
#         historical_avg = historical_avg_sales.get(store, 0)

#         if predicted_avg > historical_avg * 1.2:
#             recommendations.append(f"Store {store}: 🚀 High sales expected! Ensure sufficient inventory and staffing.")
#         elif predicted_avg < historical_avg * 0.8:
#             recommendations.append(f"Store {store}: 📉 Low sales predicted! Consider offering discounts or promotions.")
#         else:
#             recommendations.append(f"Store {store}: 📊 Sales expected to be stable. Monitor seasonal trends.")

#         # Additional insights based on external factors
#         if data[data['Store'] == store]['Holiday_Flag'].any():
#             recommendations.append(f"Store {store}: 🎉 Holiday season detected. Consider special offers or holiday-themed promotions.")

#         if data[data['Store'] == store]['Temperature'].mean() < 5:
#             recommendations.append(f"Store {store}: ❄️ Cold weather detected. Promote winter-related products.")

#         if data[data['Store'] == store]['Unemployment'].mean() > 7:
#             recommendations.append(f"Store {store}: 💼 High unemployment rate. Expect cautious spending behavior.")

#     return render_template('recommendations.html', recommendations=recommendations)