# Walmart Sales Prediction & Insights Dashboard

This project is a Flask-based web application for predicting Walmart store sales, visualizing trends, and generating actionable recommendations using AI (Google Gemini). It supports dynamic data upload, interactive dashboards, and robust error handling.

## Features

- **Data Upload:** Upload your own sales CSV files via a user-friendly interface.
- **Data Cleaning & Feature Engineering:** Automatic handling of missing values, date parsing, and feature extraction.
- **Sales Prediction:** Predicts sales using Linear Regression, Random Forest, and XGBoost models.
- **Model Evaluation:** Compares models using R², RMSE, and MAE metrics.
- **Interactive Visualizations:** Explore sales trends, store performance, and the impact of external factors (temperature, fuel price, CPI, unemployment) with Plotly charts.
- **AI-Powered Recommendations:** Integrates Google Gemini API to provide store-specific, actionable business recommendations.
- **Robust Error Handling:** User-friendly error messages for file uploads, data issues, and API failures.
- **Dashboard & Comparison:** Visual dashboards and store comparison tools for deeper insights.

## Folder Structure

```
.
├── app.py
├── Walmart.csv
├── uploads/
│   └── [uploaded CSV files]
├── static/
│   ├── [generated HTML charts]
│   └── css/
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── input.html
│   ├── results.html
│   ├── dashboard.html
│   ├── comparison.html
│   └── recommendations.html
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/walmart-sales-prediction.git
   cd walmart-sales-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Gemini API:**
   - Get your API key from Google.
   - Replace the API key in `app.py`:
     ```python
     genai.configure(api_key="YOUR_API_KEY")
     ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the app:**
   - Open your browser and go to [http://localhost:5000](http://localhost:5000)

## Usage

- **Upload Data:** Go to the Upload page and submit your sales CSV file (see required columns in the UI).
- **View Results:** After upload, view predictions, model metrics, and interactive charts.
- **Dashboard:** Explore overall trends and external factor impacts.
- **Comparison:** Compare selected stores across multiple metrics.
- **Recommendations:** Get AI-generated business recommendations for each store.

## Required CSV Columns

- Store
- Date (format: DD-MM-YYYY)
- Holiday_Flag
- Temperature
- Fuel_Price
- CPI
- Unemployment
- Weekly_Sales

## Technologies Used

- Python, Flask, Pandas, NumPy
- Scikit-learn, XGBoost
- Plotly (for interactive charts)
- Google Gemini API (for AI recommendations)
- Bootstrap (for UI)


---

**Note:** Replace sensitive information (like API keys) before sharing your
