# 📊 AI-Driven Inventory Management System

An intelligent inventory forecasting and recommendation system using AI and time-series models to optimize stock management dynamically.

---

## 🌟 Features
✔️ AI-powered material classification (Fast-Moving, Slow-Moving, Seasonal) using Azure OpenAI
✔️ Predicts future demand using ARIMA, Prophet, and Moving Average models
✔️ Recommends inventory actions based on sales forecasts
✔️ Automatically processes historical sales data from CSV
✔️ Saves forecasted results in an easy-to-read format

---

## 🚀 How It Works
1. **Load Sales Data**: Provide `mats_consumption.csv` with historical sales records.
2. **AI Classification**: The AI categorizes materials based on sales patterns.
3. **Forecasting**: The appropriate forecasting model is applied.
4. **AI Recommendations**: Inventory actions are suggested based on forecasted sales.
5. **Results**: Output is saved to `mat_forecast.csv`.

---

## 🏗️ Technologies Used
- **Python** 🐍 - Core scripting language
- **Azure OpenAI GPT-4o** 🤖 - AI-powered classification & recommendations
- **Pandas & NumPy** 📊 - Data processing
- **Statsmodels (ARIMA)** 🔢 - Time-series forecasting
- **Prophet** 📈 - Seasonal trend prediction

---

## 🔧 Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/inventory-management-ai.git
   cd inventory-management-ai
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - `ENDPOINT_URL` (Azure OpenAI endpoint)
   - `DEPLOYMENT_NAME` (Azure OpenAI model deployment name)
   - `AZURE_OPENAI_API_KEY` (Azure API key)

---



