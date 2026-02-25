import streamlit as st
import requests
import pandas as pd
import os
from dotenv import load_dotenv

# ----------------------------
# CONFIG
# ----------------------------

load_dotenv()

MONDAY_API_KEY = os.getenv("MONDAY_API_KEY")
DEALS_BOARD_ID = os.getenv("DEALS_BOARD_ID")
WORK_ORDERS_BOARD_ID = os.getenv("WORK_ORDERS_BOARD_ID")
HF_API_KEY = os.getenv("HF_API_KEY")

HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
MONDAY_URL = "https://api.monday.com/v2"

st.set_page_config(page_title="Monday BI Agent", layout="wide")
st.title("📊 Monday.com Business Intelligence Agent")
st.markdown("Founder-level AI business intelligence across Sales & Operations")

# ----------------------------
# FETCH BOARD DATA
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
        raise Exception("Failed to fetch board")

    data = response.json()
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

    status_col = find_column(df, ["deal status", "status"])
    value_col = find_column(df, ["value", "amount"])
    prob_col = find_column(df, ["probability"])

    if not status_col or not value_col:
        return 0, 0

    open_deals = df[df[status_col] != "Closed Won"]

    total_pipeline = pd.to_numeric(open_deals[value_col], errors="coerce").sum()

    if prob_col:
        weighted_pipeline = (
            pd.to_numeric(open_deals[value_col], errors="coerce")
            * pd.to_numeric(open_deals[prob_col], errors="coerce")
        ).sum()
    else:
        weighted_pipeline = total_pipeline

    return total_pipeline, weighted_pipeline


def revenue_by_sector(df):

    status_col = find_column(df, ["deal status", "status"])
    value_col = find_column(df, ["value", "amount"])
    sector_col = find_column(df, ["sector"])

    if not status_col or not value_col:
        return pd.Series()

    closed = df[df[status_col] == "Closed Won"]

    if sector_col:
        return (
            closed.groupby(sector_col)[value_col]
            .apply(lambda x: pd.to_numeric(x, errors="coerce").sum())
            .sort_values(ascending=False)
        )

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
# DASHBOARD
# ----------------------------

def build_dashboard(deals_df, work_df):

    st.subheader("📈 Executive Dashboard")

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

# ----------------------------
# HUGGINGFACE INTELLIGENCE
# ----------------------------

def interpret_query(query):

    if not HF_API_KEY:
        return rule_based_intent(query)

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    Classify this business query into one word only:
    pipeline, revenue, operations, leadership, sector, general

    Query: {query}
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 10,
            "temperature": 0.1
        }
    }

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload,
            timeout=15
        )

        if response.status_code != 200:
            return rule_based_intent(query)

        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            output = result[0]["generated_text"].lower()
            for intent in ["pipeline", "revenue", "operations", "leadership", "sector"]:
                if intent in output:
                    return intent

    except:
        return rule_based_intent(query)

    return "general"


# ----------------------------
# FALLBACK RULE ENGINE
# ----------------------------

def rule_based_intent(query):

    q = query.lower()

    if "pipeline" in q or "forecast" in q:
        return "pipeline"
    elif "revenue" in q:
        return "revenue"
    elif "operation" in q or "work order" in q:
        return "operations"
    elif "leadership" in q or "summary" in q:
        return "leadership"
    elif "sector" in q:
        return "sector"
    else:
        return "general"

# ----------------------------
# MAIN FLOW
# ----------------------------

try:
    with st.spinner("Fetching live data from monday.com..."):
        deals_df = clean_numeric_columns(fetch_board(DEALS_BOARD_ID))
        work_df = clean_numeric_columns(fetch_board(WORK_ORDERS_BOARD_ID))
except Exception:
    st.error("⚠ Unable to fetch monday.com data.")
    st.stop()

tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Chat Mode"])

with tab1:
    build_dashboard(deals_df, work_df)

with tab2:

    query = st.text_input("Ask a business question:")

    if query:
        intent = interpret_query(query)

        if intent == "pipeline":
            pipeline, weighted = calculate_pipeline(deals_df)
            st.metric("Total Pipeline", f"₹{pipeline:,.0f}")
            st.metric("Weighted Forecast", f"₹{weighted:,.0f}")

        elif intent == "revenue":
            revenue = revenue_by_sector(deals_df).sum()
            st.metric("Closed Revenue", f"₹{revenue:,.0f}")

        elif intent == "operations":
            total_orders, completed, in_progress, delayed = work_order_metrics(work_df)
            st.write(f"Total Orders: {total_orders}")
            st.write(f"Completed: {completed}")
            st.write(f"In Progress: {in_progress}")
            st.write(f"Delayed: {delayed}")

        elif intent == "leadership":
            st.markdown("### 📊 Leadership Summary")
            pipeline, weighted = calculate_pipeline(deals_df)
            revenue = revenue_by_sector(deals_df).sum()
            total_orders, completed, in_progress, delayed = work_order_metrics(work_df)

            st.write(f"Pipeline: ₹{pipeline:,.0f}")
            st.write(f"Forecast: ₹{weighted:,.0f}")
            st.write(f"Revenue: ₹{revenue:,.0f}")
            st.write(f"Work Orders: {total_orders}")
            st.write(f"Delayed: {delayed}")

        elif intent == "sector":
            st.bar_chart(revenue_by_sector(deals_df))

        else:
            st.write("Could you clarify your request?")
