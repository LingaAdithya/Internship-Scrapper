from datetime import datetime, timezone
import pandas as pd
from jobspy import scrape_jobs
import firebase_admin
from firebase_admin import credentials, firestore
import os
import requests
from bs4 import BeautifulSoup
import re
import time

# --- Firebase Setup ---
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Search Terms ---
search_terms = [
    "software intern", "summer internship", "internship", "Software Internship",
    "Internships", "Paid Internship", "Winter Internship",
    "AI Intern", "Machine Learning Intern", "Data Science Intern", "Web Developer Intern"
]

def clean_text(text):
    """Clean and normalize text."""
    return re.sub(r'\s+', ' ', text).strip()

def extract_full_description(url):
    """Fetch the full description from the job page."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Try to extract the main content block
        paragraphs = soup.find_all(["p", "li"])
        text = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
        return clean_text(text) if len(text) > 100 else None
    except Exception as e:
        print(f"⚠️ Failed to fetch description from {url}: {e}")
        return None

def update_internships():
    for term in search_terms:
        print(f"\n🔍 Scraping internships for: {term}")
        try:
            jobs = scrape_jobs(
                site_name=["linkedin", "google"],
                search_term=term,
                google_search_term=f"{term} in India",
                location="India",
                results_wanted=100,
                hours_old=168,  # last 7 days
                country_indeed='INDIA'
            )

            jobs_df = pd.DataFrame(jobs)
            if jobs_df.empty:
                print(f"⚠️ No jobs found for {term}")
                continue

            for _, row in jobs_df.iterrows():
                title = row.get("title")
                company_name = row.get("company_name") or row.get("company") or ""
                description = row.get("description") or row.get("qualifications") or row.get("benefits") or ""
                link = row.get("job_url")
                source = row.get("site")

                # If description is missing, fetch from job URL
                if not description or len(description) < 50:
                    description = extract_full_description(link) or "No description available."

                # Prevent duplicate inserts
                existing = db.collection("internships").where("link", "==", link).get()

                data = {
                    "title": title,
                    "company_name": company_name,
                    "description": description,
                    "link": link,
                    "source": source,
                    "deadline": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }

                if not existing:
                    db.collection("internships").add(data)
                    print(f"✅ Added: {title} at {company_name}")
                else:
                    doc_id = existing[0].id
                    db.collection("internships").document(doc_id).update(data)
                    print(f"🔄 Updated: {title} at {company_name}")

                # Be polite with requests
                time.sleep(0.5)

        except Exception as e:
            print(f"❌ Error scraping {term}: {e}")

if __name__ == "__main__":
    update_internships()
