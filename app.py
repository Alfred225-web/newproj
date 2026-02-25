import streamlit as st
import requests
import pandas as pd
from openai import OpenAI

# ----------------------------
# CONFIG (STREAMLIT CLOUD)
# ----------------------------

MONDAY_API_KEY = st.secrets["MONDAY_API_KEY"].strip()
DEALS_BOARD_ID = st.secrets["DEALS_BOARD_ID"]
WORK_ORDERS_BOARD_ID = st.secrets["WORK_ORDERS_BOARD_ID"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Groq via OpenAI SDK
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MONDAY_URL = "https://api.monday.com/v2"

st.set_page_config(page_title="Monday BI Agent", layout="wide")
st.title("📊 Monday.com Business Intelligence Agent")

# ----------------------------
# FETCH BOARD FROM MONDAY
# ----------------------------
def fetch_board(board_id):

    # Safety checks
    if not board_id:
        raise Exception("Board ID is missing.")

    if not MONDAY_API_KEY:
        raise Exception("Monday API key is missing.")

    # Ensure board ID is integer
    try:
        board_id = int(board_id)
    except:
        raise Exception("Board ID must be numeric.")

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

    headers = {
        "Authorization": MONDAY_API_KEY.strip(),
        "Content-Type": "application/json"
    }

    response = requests.post(
        MONDAY_URL,
        json={"query": query},
        headers=headers
    )

    # Debugging (remove later if you want)
    if response.status_code != 200:
        raise Exception(f"HTTP Error {response.status_code}: {response.text}")

    data = response.json()

    if "errors" in data:
        raise Exception(f"Monday API Error: {data['errors']}")

    boards = data.get("data", {}).get("boards", [])

    if not boards:
        raise Exception("Board not found. Check Board ID.")

    items = boards[0].get("items_page", {}).get("items", [])

    if not items:
        return pd.DataFrame()  # Return empty safely

    rows = []

    for item in items:
        row = {"Item Name": item.get("name", "")}

        for col in item.get("column_values", []):
            col_title = col.get("column", {}).get("title", "")
            row[col_title] = col.get("text", "")

        rows.append(row)

    return pd.DataFrame(rows)

# ----------------------------
# HELPERS
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

    total_pipeline = pd.to_numeric(open_deals[value_col], errors="coerce").sum()

    if prob_col:
        weighted = (
            pd.to_numeric(open_deals[value_col], errors="coerce") *
            pd.to_numeric(open_deals[prob_col], errors="coerce")
        ).sum()
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

    total = len(df)

    if not status_col:
        return total, 0, 0, 0

    counts = df[status_col].value_counts()

    return (
        total,
        counts.get("Completed", 0),
        counts.get("In Progress", 0),
        counts.get("Delayed", 0)
    )

# ----------------------------
# AI INTENT (GROQ)
# ----------------------------

def interpret_query(query):

    prompt = f"""
    Classify this query into one of:
    pipeline
    revenue
    operations
    leadership
    sector
    general

    Query: {query}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.lower()

# ----------------------------
# MAIN APP
# ----------------------------

try:
    with st.spinner("Fetching live data from Monday.com..."):
        deals_df = clean_numeric_columns(fetch_board(DEALS_BOARD_ID))
        work_df = clean_numeric_columns(fetch_board(WORK_ORDERS_BOARD_ID))
except Exception as e:
    st.error(str(e))
    st.stop()

tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Chat Mode"])

# ----------------------------
# DASHBOARD TAB
# ----------------------------

with tab1:

    pipeline, weighted = calculate_pipeline(deals_df)
    revenue = revenue_by_sector(deals_df).sum()
    total_orders, completed, in_progress, delayed = work_order_metrics(work_df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Pipeline", f"₹{pipeline:,.0f}")
    col2.metric("Weighted Forecast", f"₹{weighted:,.0f}")
    col3.metric("Closed Revenue", f"₹{revenue:,.0f}")
    col4.metric("Work Orders", total_orders)

# ----------------------------
# CHAT TAB
# ----------------------------

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
            st.write("Leadership overview:")
            st.write(f"Pipeline: ₹{pipeline:,.0f}")
            st.write(f"Revenue: ₹{revenue:,.0f}")
            st.write(f"Work Orders: {total_orders}")

        elif "sector" in intent:
            st.bar_chart(revenue_by_sector(deals_df))

        else:
            st.write("Could you clarify your request?")


