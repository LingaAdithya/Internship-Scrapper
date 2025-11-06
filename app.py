import os
import json
import time
import fitz  # PyMuPDF
import torch
import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from vertexai.preview.generative_models import GenerativeModel
import vertexai
from math import ceil
from sentence_transformers import SentenceTransformer, util
import os

from dotenv import load_dotenv
import os

load_dotenv()

# ==============================
# 🔧 BASIC CONFIG
# ==============================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
MODEL_NAME =  os.getenv("MODEL_NAME")

# ==============================
# 🧩 FIREBASE INITIALIZATION
# ==============================
@st.cache_resource(show_spinner=False)
def init_firestore():
    """Safely initialize Firebase app and Firestore client."""
    try:
        firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate("firebase_credentials.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firestore()

# ==============================
# 🧠 VERTEX AI MODEL SETUP
# ==============================
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(MODEL_NAME)

# ==============================
# ⚡ FAST EMBEDDER CACHING
# ==============================
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

embedder = load_embedder()

# ==============================
# 📄 PDF TEXT EXTRACTION
# ==============================
def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text.strip()

# ==============================
# 🔥 FETCH INTERNSHIPS
# ==============================
def get_internships():
    """Fetch internships from Firestore and attach document IDs."""
    docs = db.collection("internships").order_by("created_at", direction=firestore.Query.DESCENDING).stream()
    data = []
    for d in docs:
        item = d.to_dict()
        item["id"] = d.id  # Attach Firestore doc ID
        data.append(item)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce")
    df = df.sort_values(by="created_at", ascending=False)
    return df


# ==============================
# 🚀 FAST + EFFICIENT MATCHING
# ==============================
def get_best_matches(resume_text, internships, top_k_filter=30):
    """
    Step 1: Filter internships by semantic similarity using embeddings (fast, local)
    Step 2: Send only top-K relevant ones to Gemini for detailed ranking
    """
    if not internships:
        return []

    internship_texts = [
        (i.get("id"), f"{i.get('title', '')} {i.get('company_name', '')} {i.get('description', '')}")
        for i in internships
    ]
    internship_ids, corpus = zip(*internship_texts)

    resume_emb = embedder.encode(resume_text, convert_to_tensor=True)
    corpus_emb = embedder.encode(corpus, convert_to_tensor=True)

    scores = util.cos_sim(resume_emb, corpus_emb)[0]
    top_results = torch.topk(scores, k=min(top_k_filter, len(scores)))

    filtered_internships = [internships[i.item()] for i in top_results.indices]

    # --- Send to Gemini ---
    job_data = "\n".join([
        f"Job ID: {i.get('id', 'N/A')}\n"
        f"Title: {i.get('title', 'N/A')}\n"
        f"Company: {i.get('company_name', 'N/A')}\n"
        f"Description: {i.get('description', 'N/A')}\n"
        f"Link: {i.get('link', 'N/A')}\n"
        for i in filtered_internships
    ])

    prompt = f"""
    You are the head of human resource with specialized training to recommend the best internships for a given resume.
    Given the resume and internship list below, return the TOP 5 matches with a crisp and brutally honest reasoning and match score.

    Resume:
    {resume_text}

    Internships:
    {job_data}

    Respond ONLY in JSON array format like:
    [
      {{
        "job_id": "<job id>",
        "match_score": <strict evaluation score out of 100, by matching the job's requirements with the resume>,
        "reason": "<A valid, and a concise reason explaining why this job is a good match based on the resume>"
      }},
      ...
    ]
    """

    response = model.generate_content(prompt)

    try:
        text = response.text.strip()
        if text.startswith("```"):
            text = text.strip("```json").strip("```")
        return json.loads(text)
    except Exception:
        return []  # silent fail, no raw output

# ==============================
# 🎨 STREAMLIT UI
# ==============================
st.set_page_config(page_title="AI Internship Recommender", page_icon="🎯", layout="wide")

st.title("🎯 AI-Powered Internship Recommender")
st.write("Upload your resume and let the fine-tuned Gemini model find your best-fit internships.")

# Resume Upload
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)

    st.subheader("📄 Resume Preview")
    st.text_area("Extracted Resume Text", resume_text[:] + "...", height=200)

    if st.button("🔍 Find My Top 5 Matches"):
        with st.spinner("Analyzing and matching..."):
            internships = get_internships().to_dict(orient="records")
            matches = get_best_matches(resume_text, internships)

        if matches:
            all_jobs_df = pd.DataFrame(internships)
            merged = pd.DataFrame(matches).merge(all_jobs_df, left_on="job_id", right_on="id", how="left")
            merged = merged.sort_values(by="match_score", ascending=False).reset_index(drop=True)

            st.subheader("🏆 Top 5 Recommended Internships")

            for i, row in merged.iterrows():
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #007BFF33;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 12px;
                        background-color: #f9fbff;">
                        <h4 style="margin-bottom:5px;">
                            <b>#{i+1}. {row.get('title', 'N/A')}</b>
                            <span style="float:right;color:#007BFF;">Match Score: {row.get('match_score', 0)}%</span>
                        </h4>
                        <p style="margin:5px 0 10px;color:#444;">
                            <b>Company:</b> {row.get('company_name', 'N/A')}
                        </p>
                        <p style="margin:0 0 10px;color:#555;">
                            <b>Reason:</b> {row.get('reason', 'No reason available.')}
                        </p>
                        <a href="{row.get('link', '#')}" target="_blank">
                            <button style="background-color:#007BFF;color:white;
                                border:none;border-radius:6px;
                                padding:6px 14px;cursor:pointer;">
                                Apply
                            </button>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("No valid matches found. Try again later or check your model setup.")

# ==============================
# 📚 ALL INTERNSHIPS TABLE
# ==============================
st.subheader("📚 All Available Internships")

df = get_internships()
if df.empty:
    st.info("No internships found in the database.")
else:
    per_page = 30
    total_pages = ceil(len(df) / per_page)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_df = df.iloc[start_idx:end_idx]

    paginated_df = paginated_df[["title", "company_name", "link"]].rename(
        columns={"title": "Title", "company_name": "Company"}
    )
    paginated_df["Apply"] = paginated_df["link"].apply(
        lambda x: f'<a href="{x}" target="_blank">'
                  f'<button style="background-color:#007BFF;color:white;'
                  f'border:none;border-radius:6px;padding:6px 12px;cursor:pointer;">Apply</button></a>'
    )
    paginated_df.drop(columns=["link"], inplace=True)

    st.markdown(
        """
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
                font-family: 'Inter', sans-serif;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #33333322;
            }
            th {
                background-color: #007BFF15;
                color: #007BFF;
                font-weight: 600;
            }
            tr:hover {
                background-color: #007BFF08;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        paginated_df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

    st.markdown(f"<p style='text-align:center;'>Page {page} of {total_pages}</p>", unsafe_allow_html=True)
