import streamlit as st
import pandas as pd
from supabase import create_client, Client
import os

# --- Supabase Setup ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="Internship Opportunities", page_icon="🎯", layout="wide")

st.title("🎯 Internship Opportunities")

# --- Fetch Data from Supabase ---
@st.cache_data(ttl=300)  # cache results for 5 minutes
def fetch_internships():
    response = supabase.table("internships").select("*").order("created_at", desc=True).execute()
    return pd.DataFrame(response.data)

df = fetch_internships()

if df.empty:
    st.warning("No internships found in the database.")
else:
    # --- Filters ---
    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("🔍 Search internships", "")
    with col2:
        source_filter = st.multiselect("Filter by Source", df["source"].dropna().unique())

    filtered_df = df.copy()

    # Apply search
    if search:
        filtered_df = filtered_df[
            filtered_df["title"].str.contains(search, case=False, na=False) |
            filtered_df["company_name"].str.contains(search, case=False, na=False) |
            filtered_df["description"].str.contains(search, case=False, na=False)
        ]

    # Apply source filter
    if source_filter:
        filtered_df = filtered_df[filtered_df["source"].isin(source_filter)]

    # --- Display in Table ---
    st.write(f"Showing **{len(filtered_df)}** internships")
    st.dataframe(
        filtered_df[["title", "company_name", "source", "link", "description", "created_at"]],
        use_container_width=True,
        hide_index=True
    )

    # --- Expandable Cards for Details ---
    st.markdown("### 📋 Internship Details")
    for _, row in filtered_df.iterrows():
        with st.expander(f"{row['title']} — {row['company_name']}"):
            st.markdown(f"**Company:** {row['company_name'] or 'N/A'}")
            st.markdown(f"**Source:** {row['source'] or 'N/A'}")
            if row["link"]:
                st.markdown(f"🔗 [Apply Here]({row['link']})")
            st.write(row["description"] or "No description available.")
