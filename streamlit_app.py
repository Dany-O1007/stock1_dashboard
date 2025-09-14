pip install openai
pip install streamlit openai requests
import streamlit as st
import pandas as pd
import math
import csv
from openai import OpenAI
from pathlib import Path

# ---------------------- Configuration ----------------------
# DO NOT expose API keys in plain input fields.
# Use a secure variable or secret for deployment.
API_KEY: ${{ secrets.My_Open_API }}

# ---------------------- App UI Setup ----------------------
st.set_page_config(page_title="📈 Stock News Dashboard", layout="wide")
st.title("📈 AI-Powered Stock News Dashboard")

# ---------------------- Functions ----------------------

# Fetch latest stock news and save to CSV
def fetch_stock_news():
    URL = "https://elite.finviz.com/news_export.ashx?v=36"
    response = requests.get(URL)
    if response.status_code == 200:
        with open("export.csv", "wb") as file:
            file.write(response.content)
        return "✅ News data successfully saved to export.csv"
    else:
        return "❌ Failed to fetch news data."

# Analyze a single news article with OpenAI
def analyze_news_article(title, stock, api_key):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": f"Analyze the article titled: '{title}'. Give a clear, concise dashboard recommendation for {stock} on whether to buy, sell, or hold."}
        ]
    )

    return response.choices[0].message.content.strip()

# Process all news data and return analysis
def process_news_data(api_key):
    recommendations = []
    try:
        with open("export.csv", "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    title = row[0]
                    stock = row[1]
                    recommendation = analyze_news_article(title, stock, api_key)
                    recommendations.append((title, stock, recommendation))
        return recommendations
    except FileNotFoundError:
        st.error("⚠️ export.csv not found. Please fetch stock news first.")
        return []

# ---------------------- Streamlit Layout ----------------------

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("📥 Fetch Latest News"):
        result = fetch_stock_news()
        st.success(result)

    if st.button("📊 Generate AI Recommendations"):
        st.info("Analyzing news with GPT-4o... Please wait ⏳")
        with st.spinner("Running analysis..."):
            recommendations = process_news_data(API_KEY)

        if recommendations:
            with col2:
                st.subheader("💡 AI Investment Suggestions")
                for title, stock, recommendation in recommendations:
                    with st.expander(f"{stock} — {title[:80]}"):
                        st.markdown(f"**📌 Recommendation:**\n\n{recommendation}")
        else:
            st.warning("No recommendations found. Please fetch news first.")

streamlit run streamlit_app.py
