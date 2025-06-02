import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="Prediction History", layout="centered")

st.title("📜 Prediction History")

# Connect to DB
conn = sqlite3.connect("database/predictions.db", check_same_thread=False)
c = conn.cursor()

# Load data
df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)

if not df.empty:
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(df[["filename", "result", "confidence", "timestamp"]])
else:
    st.info("No predictions found. Make your first prediction on the Home page.")
