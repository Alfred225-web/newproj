import streamlit as st
import requests
import pandas as pd
import os

from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------



MONDAY_API_KEY = st.secrets["MONDAY_API_KEY"]
DEALS_BOARD_ID = st.secrets["DEALS_BOARD_ID"]
WORK_ORDERS_BOARD_ID = st.secrets["WORK_ORDERS_BOARD_ID"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

MONDAY_URL = "https://api.monday.com/v2"

client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1"
)

st.set_page_config(page_title="Monday BI Agent", layout="wide")
st.title("📊 Monday.com Business Intelligence Agent")
st.markdown("Founder-level AI business intelligence across Sales & Operations")

# ----------------------------
# FETCH BOARD
# ----------------------------

def fetch_board(board_id):

    query = f"""
    {{
      boards(ids: {board_id}) {{
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

    headers = {"Authorization": MONDAY_API_KEY}
    response = requests.post(MONDAY_URL, json={"query": query}, headers=headers)

    if response.status_code != 200:
        st.error("Failed to fetch board from Monday.")
        st.stop()

    data = response.json()

    if not data.get("data") or not data["data"]["boards"]:
        st.error("Board not found or API key invalid.")
        st.stop()

    items = data["data"]["boards"][0]["items_page"]["items"]

    rows = []
    for item in items:
        row = {"Item Name": item["name"]}
        for col in item["column_values"]:
            row[col["column"]["title"]] = col["text"]
        rows.append(row)

    return pd.DataFrame(rows)

# ----------------------------
# UTILITIES
# ----------------------------

def find_column(df, keywords):
    for keyword in keywords:
        for col in df.columns:
            if keyword.lower() in col.lower():
                return col
    return None


def clean_numeric_columns(df):
    for col in df.columns:
        if any(word in col.lower() for word in ["value", "amount", "probability"]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ----------------------------
# SALES LOGIC
# ----------------------------

def calculate_pipeline(df):

    status_col = find_column(df, ["status"])
    value_col = find_column(df, ["value"])
    prob_col = find_column(df, ["probability"])

    if not status_col or not value_col:
        return 0, 0

    open_deals = df[df[status_col] != "Closed Won"]

    total_pipeline = open_deals[value_col].sum()

    if prob_col:
        weighted = (open_deals[value_col] * open_deals[prob_col]).sum()
    else:
        weighted = total_pipeline

    return total_pipeline, weighted


def revenue_by_sector(df):

    status_col = find_column(df, ["status"])
    value_col = find_column(df, ["value"])
    sector_col = find_column(df, ["sector"])

    if not status_col or not value_col:
        return pd.Series()

    closed = df[df[status_col] == "Closed Won"]

    if sector_col:
        return closed.groupby(sector_col)[value_col].sum().sort_values(ascending=False)

    return pd.Series()

# ----------------------------
# OPERATIONS LOGIC
# ----------------------------

def work_order_metrics(df):

    status_col = find_column(df, ["status"])
    total_orders = len(df)

    if not status_col:
        return total_orders, 0, 0, 0

    status_counts = df[status_col].value_counts()

    completed = status_counts.get("Completed", 0)
    in_progress = status_counts.get("In Progress", 0)
    delayed = status_counts.get("Delayed", 0)

    return total_orders, completed, in_progress, delayed

# ----------------------------
# AI INTENT CLASSIFIER (GROK)
# ----------------------------

def interpret_query(query):

    prompt = f"""
Classify this business question into ONE word only:

pipeline
revenue
operations
leadership
sector
general

Question: {query}

Respond with only one word.
"""

    response = client.chat.completions.create(
        model="grok-2-latest",
        messages=[
            {"role": "system", "content": "You are a business analytics classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip().lower()

# ----------------------------
# MAIN APP
# ----------------------------

try:
    with st.spinner("Fetching live data from monday.com..."):
        deals_df = clean_numeric_columns(fetch_board(DEALS_BOARD_ID))
        work_df = clean_numeric_columns(fetch_board(WORK_ORDERS_BOARD_ID))

except Exception as e:
    st.error("Unable to fetch Monday data.")
    st.stop()

tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Chat Mode"])

with tab1:

    pipeline, weighted = calculate_pipeline(deals_df)
    revenue = revenue_by_sector(deals_df).sum()
    total_orders, completed, in_progress, delayed = work_order_metrics(work_df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Pipeline", f"₹{pipeline:,.0f}")
    col2.metric("Weighted Forecast", f"₹{weighted:,.0f}")
    col3.metric("Closed Revenue", f"₹{revenue:,.0f}")
    col4.metric("Work Orders", total_orders)

    sector_data = revenue_by_sector(deals_df)
    if not sector_data.empty:
        st.subheader("Revenue by Sector")
        st.bar_chart(sector_data)

with tab2:

    query = st.text_input("Ask a business question:")

    if query:

        intent = interpret_query(query)

        if "pipeline" in intent:
            pipeline, weighted = calculate_pipeline(deals_df)
            st.metric("Total Pipeline", f"₹{pipeline:,.0f}")
            st.metric("Weighted Forecast", f"₹{weighted:,.0f}")

        elif "revenue" in intent:
            revenue = revenue_by_sector(deals_df).sum()
            st.metric("Closed Revenue", f"₹{revenue:,.0f}")

        elif "operations" in intent:
            total_orders, completed, in_progress, delayed = work_order_metrics(work_df)
            st.write(f"Total Orders: {total_orders}")
            st.write(f"Completed: {completed}")
            st.write(f"In Progress: {in_progress}")
            st.write(f"Delayed: {delayed}")

        elif "leadership" in intent:
            st.write("Leadership summary coming soon.")

        elif "sector" in intent:
            st.bar_chart(revenue_by_sector(deals_df))

        else:
            st.write("Could you clarify your request?")