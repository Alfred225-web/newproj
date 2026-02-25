
import streamlit as st
import requests
import pandas as pd
from openai import OpenAI

# =========================
# CONFIG (Streamlit Secrets)
# =========================


MONDAY_API_KEY = st.secrets["MONDAY_API_KEY"]
DEALS_BOARD_ID = int(st.secrets["DEALS_BOARD_ID"])
WORK_ORDERS_BOARD_ID = int(st.secrets["WORK_ORDERS_BOARD_ID"])
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

MONDAY_URL = "https://api.monday.com/v2"
st.write("MONDAY KEY EXISTS:", bool(st.secrets.get("MONDAY_API_KEY")))
# Groq Client (OpenAI compatible)
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(page_title="Monday BI Agent", layout="wide")
st.title("📊 Monday.com Business Intelligence Agent")
st.markdown("Founder-level AI business intelligence across Sales & Operations")

# =========================
# FETCH BOARD FROM MONDAY
# =========================

def fetch_board(board_id):

    query = f"""
    {{
      boards(ids: [{board_id}]) {{
        id
        name
        items_page(limit: 500) {{
          items {{
            name
            column_values {{
              text
              column {{
                title
              }}
            }}
          }}
        }}
      }}
    }}
    """

    headers = {
        "Authorization": MONDAY_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(
        MONDAY_URL,
        json={"query": query},
        headers=headers
    )

    data = response.json()

    if response.status_code != 200:
        st.error("Monday API request failed.")
        st.write(data)
        st.stop()

    if "errors" in data:
        st.error("Monday GraphQL Error:")
        st.write(data["errors"])
        st.stop()

    boards = data.get("data", {}).get("boards", [])

    if not boards:
        st.error("Board not found or no access.")
        st.stop()

    items = boards[0]["items_page"]["items"]

    rows = []
    for item in items:
        row = {"Item Name": item["name"]}
        for col in item["column_values"]:
            row[col["column"]["title"]] = col["text"]
        rows.append(row)

    return pd.DataFrame(rows)

# =========================
# CLEANING
# =========================

def clean_numeric(df):
    for col in df.columns:
        if any(word in col.lower() for word in ["value", "amount", "probability"]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# =========================
# SALES METRICS
# =========================

def calculate_pipeline(df):

    status_col = next((c for c in df.columns if "status" in c.lower()), None)
    value_col = next((c for c in df.columns if "value" in c.lower()), None)
    prob_col = next((c for c in df.columns if "probability" in c.lower()), None)

    if not status_col or not value_col:
        return 0, 0

    open_deals = df[df[status_col] != "Closed Won"]

    total = pd.to_numeric(open_deals[value_col], errors="coerce").sum()

    if prob_col:
        weighted = (
            pd.to_numeric(open_deals[value_col], errors="coerce")
            * pd.to_numeric(open_deals[prob_col], errors="coerce")
        ).sum()
    else:
        weighted = total

    return total, weighted

# =========================
# AI QUERY CLASSIFIER
# =========================

def interpret_query(query):

    prompt = f"""
Classify this into ONE WORD only:

pipeline
revenue
operations
leadership
sector
general

Question: {query}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You classify business questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip().lower()

# =========================
# LOAD DATA
# =========================

with st.spinner("Fetching live data from Monday.com..."):
    deals_df = clean_numeric(fetch_board(DEALS_BOARD_ID))
    work_df = clean_numeric(fetch_board(WORK_ORDERS_BOARD_ID))

# =========================
# DASHBOARD
# =========================

st.subheader("📈 Executive Dashboard")

pipeline, weighted = calculate_pipeline(deals_df)

col1, col2 = st.columns(2)

col1.metric("Total Pipeline", f"₹{pipeline:,.0f}")
col2.metric("Weighted Forecast", f"₹{weighted:,.0f}")

# =========================
# CHAT MODE
# =========================

st.subheader("🤖 Chat Mode")

query = st.text_input("Ask a business question:")

if query:

    intent = interpret_query(query)

    if "pipeline" in intent:
        st.metric("Total Pipeline", f"₹{pipeline:,.0f}")
        st.metric("Weighted Forecast", f"₹{weighted:,.0f}")
    else:
        st.write("Try asking about pipeline.")


