import os
import pandas as pd
import numpy as np
from openai import AzureOpenAI
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Azure OpenAI Client Setup
endpoint = os.getenv("ENDPOINT_URL", "REPLACE YOUR ENDPOINT")
deployment = os.getenv("DEPLOYMENT_NAME", "REPLACEYOUR DEPLOYMENT")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE YOUR API KEY")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="REPALCE YOUR VERSION",
)

class AIDrivenClassificationAgent:
    """AI agent to classify materials based on sales data."""
    def classify_material(self, sales_data):
        chat_prompt = [
            {"role": "system", "content": [{"type": "text", "text": "You are an AI assistant specializing in inventory classification and demand forecasting."}]},
            {"role": "user", "content": [{"type": "text", "text": f"""
                Analyze the following material data and classify it into one of the following categories:
                - Fast-Moving
                - Slow-Moving
                - Seasonal
                **Sales Data (Last 3 Years - Monthly):** {list(sales_data)}
                Provide ONLY the category name in response. No explanations.
            """}]}]

        completion = client.chat.completions.create(
            model=deployment, messages=chat_prompt, max_tokens=5, temperature=0.3, top_p=0.9
        )
        return completion.choices[0].message.content.strip()

class MovingAverageForecastAgent:
    """Forecasting agent using moving averages."""
    def forecast(self, sales_data):
        if len(sales_data) < 6:
            return [max(0, round(np.mean(sales_data)))] * 6  
        ma_forecast = pd.Series(sales_data).rolling(window=12, min_periods=6).mean().dropna().values[-1]
        return [max(0, round(ma_forecast))] * 6 

class SlowMovingForecastAgent:
    """Forecasting agent using ARIMA for slow-moving materials."""
    def forecast(self, sales_data):
        if len(sales_data) < 12:
            return [max(0, round(np.mean(sales_data)))] * 6 
        model = ARIMA(sales_data, order=(5, 1, 0))
        model_fit = model.fit()
        return [max(0, round(value)) for value in model_fit.forecast(steps=6)]

class SeasonalForecastAgent:
    """Forecasting agent using Facebook Prophet for seasonal materials."""
    def forecast(self, sales_data):
        if len(sales_data) < 5:
            return [max(0, round(np.mean(sales_data)))] * 6  
        df = pd.DataFrame({"ds": pd.date_range(start="2024-01-01", periods=len(sales_data), freq="M"), "y": sales_data})
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=6, freq="M")
        forecast_df = model.predict(future)
        return [max(0, round(value)) for value in forecast_df["yhat"].tail(6)]

class AIRecommendationAgent:
    """AI agent to determine inventory action based on forecast data."""
    def decide_action(self, forecast_data, category):
        chat_prompt = [
            {"role": "system", "content": [{"type": "text", "text": "You are an AI assistant specializing in inventory decision-making."}]},
            {"role": "user", "content": [{"type": "text", "text": f"""
                - **Forecasted Sales for Next 6 Months:** {forecast_data}
                - **Material Classification:** {category}
                Recommend the most suitable inventory action for the next 6 months. Provide only the recommended action, without explanations.
            """}]}]
        
        completion = client.chat.completions.create(
            model=deployment, messages=chat_prompt, max_tokens=10, temperature=0.3
        )
        return completion.choices[0].message.content.strip()

class AIInventoryManagementSystem:
    """Main AI-driven inventory forecasting and decision-making system."""
    def run(self, sales_data):
        classification_agent = AIDrivenClassificationAgent()
        decision_agent = AIRecommendationAgent()
        
        category = classification_agent.classify_material(sales_data)
        if "fast" in category.lower():
            forecast = MovingAverageForecastAgent().forecast(sales_data)
        elif "slow" in category.lower():
            forecast = SlowMovingForecastAgent().forecast(sales_data)
        else:
            forecast = SeasonalForecastAgent().forecast(sales_data)
        
        decision = decision_agent.decide_action(forecast, category)
        return {"category": category, "forecast": forecast, "decision": decision}

# Load Data
df = pd.read_csv("mats_consumption.csv")
df["MONTH"] = pd.to_datetime(df["MONTH"])
forecast_months = pd.date_range(start="2025-01-01", periods=6, freq="M")
forecast_data = []

# Process Forecasting for Each Material
for material in df["MATERIAL"].unique():
    sales_data = df[df["MATERIAL"] == material]["SALES"].values
    result = AIInventoryManagementSystem().run(sales_data)
    for i in range(6):
        forecast_data.append({
            "MONTH": forecast_months[i], "MATERIAL": material,
            "Category": result["category"], "Forecast": result["forecast"][i], "Decision": result["decision"]
        })

# Save Results
df_forecast = pd.DataFrame(forecast_data)
df = pd.concat([df, df_forecast], ignore_index=True)
df.to_csv("mat_forecast.csv", index=False)
print("✅ Forecasting completed! Results saved to 'mat_forecast.csv'.")
