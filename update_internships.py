from datetime import datetime, timezone
import pandas as pd
from jobspy import scrape_jobs
from supabase import create_client, Client
import datetime
import os

# --- Supabase Setup ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- List of Search Terms ---
search_terms = [
    "software intern",
    "summer internship",
    "internship",
    "Software Internship",
]

def update_internships():
    for term in search_terms:
        print(f"Scraping internships for: {term}")
        try:
            jobs = scrape_jobs(
                site_name=["linkedin", "google"],  
                search_term=term,
                google_search_term=f"{term} in India",
                location="India",
                results_wanted=200,
                hours_old=168,  # last 7 days
                country_indeed='INDIA'
            )

            jobs_df = pd.DataFrame(jobs)
            if jobs_df.empty:
                print(f"No jobs found for {term}")
                continue

            for _, row in jobs_df.iterrows():
                title = row.get("title")
                company_name = row.get("company_name") or row.get("company") or ""
                description = row.get("description") or ""
                link = row.get("job_url")
                source = row.get("site")
                deadline = None

                # Check if job already exists
                existing = supabase.table("internships").select("id").eq("link", link).execute()
                if len(existing.data) == 0:
                    # Insert new job
                    supabase.table("internships").insert({
                        "title": title,
                        "company_name": company_name,
                        "description": description,
                        "deadline": deadline,
                        "link": link,
                        "source": source,
                        "created_at": datetime.datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.datetime.now(timezone.utc).isoformat()
                    }).execute()
                else:
                    # Update existing job
                    supabase.table("internships").update({
                        "title": title,
                        "company_name": company_name,
                        "description": description,
                        "updated_at": datetime.datetime.now(timezone.utc).isoformat()
                    }).eq("link", link).execute()
        except Exception as e:
            print(f"Error scraping {term}: {e}")

if __name__ == "__main__":
    update_internships()