import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import random

# =========================
# Load Data and Models
# =========================
df = pickle.load(open("job_data.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# =========================
# Data Cleanup
# =========================
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

job_types = ["Permanent Full-time", "Part-time", "Work From Home", "On-site", "Contract"]
if "job_type" not in df.columns:
    df["job_type"] = [random.choice(job_types) for _ in range(len(df))]
else:
    df["job_type"] = df["job_type"].fillna(pd.Series([random.choice(job_types) for _ in range(len(df))]))

df["avg_hourly_rate"] = df["avg_hourly_rate"].fillna(0)
df["country"] = df["country"].fillna("Unknown")

# =========================
# Salary Formatting Function
# =========================
def format_salary(rate, country, experience_level):
    """Format salary in INR or USD, adjusted by experience level."""
    multiplier = 1.0
    if experience_level == "Fresher":
        multiplier = 0.8
    elif experience_level == "Mid-Level":
        multiplier = 1.2
    elif experience_level == "Senior":
        multiplier = 1.5

    rate = rate * multiplier

    if "india" in country.lower():
        if rate >= 100000:
            return f"₹{rate/100000:.1f} Lakh"
        elif rate >= 1000:
            return f"₹{rate/1000:.1f} Thousand"
        else:
            return f"₹{rate:.2f}/hr"
    else:
        if rate >= 100000:
            return f"${rate/100000:.1f}K"
        elif rate >= 1000:
            return f"${rate/1000:.1f}K"
        else:
            return f"${rate:.2f}/hr"

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="Job Market Recommendation System",
    page_icon="👩‍💻",
    layout="wide"
)

# =========================
# Custom CSS Styling
# =========================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #fff8d6, #fff5b0);
            font-family: "Segoe UI", sans-serif;
            color: #2c3e50;
        }
        .main-title {
            font-size: 44px;
            font-weight: 850;
            color: #2e2e2e;
            text-align: center;
            text-shadow: 1px 1px 3px #ffcc00;
            margin-bottom: 5px;
        }
        .section-header {
            font-size: 26px;
            font-weight: 750;
            color: #ff8c00;
            margin-top: 35px;
            margin-bottom: 10px;
        }
        .stDataFrame {
            font-weight: 600;
        }
        .stTextInput > div > div > input {
            border: 2px solid #ffcc00;
            border-radius: 10px;
        }
        .stSelectbox > div > div {
            border: 2px solid #ffcc00;
            border-radius: 10px;
        }
        .stTextArea textarea {
            border: 2px solid #ffcc00;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #ffcc33;
            color: black;
            font-weight: bold;
            border-radius: 12px;
            border: none;
            padding: 8px 20px;
        }
        .stButton>button:hover {
            background-color: #ffb700;
        }
        table td, table th {
            text-align: center !important;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# App Title
# =========================
st.markdown('<h1 class="main-title"> 👩‍💻🏢👨‍💻 Job Market Analysis & Recommendation System</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:17px;'>Discover global job trends, compare real salaries, and find your perfect career fit!</p>", unsafe_allow_html=True)

# =========================
# Job Recommendation Engine
# =========================
st.markdown('<h2 class="section-header">🔍 Job Recommendation Engine</h2>', unsafe_allow_html=True)
job_input = st.text_input("Enter job title or skill:")

if job_input:
    tfidf_matrix = vectorizer.transform(df["title"].astype(str))
    query_vec = vectorizer.transform([job_input])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = similarity.argsort()[-5:][::-1]
    results = df.iloc[top_idx][["title", "avg_hourly_rate", "country", "job_type"]]
    results["Formatted Salary"] = results.apply(lambda x: format_salary(x["avg_hourly_rate"], x["country"], "Mid-Level"), axis=1)

    st.write("### 🏆 Top Recommended Jobs")
    st.dataframe(results.reset_index(drop=True))
else:
    st.info("Enter a job title or skill to get personalized job recommendations.")

# =========================
# Filter Jobs by Country
# =========================
st.markdown('<h2 class="section-header">🌎 Filter Jobs by Country</h2>', unsafe_allow_html=True)
unique_countries = sorted(df["country"].dropna().unique())
country_select = st.selectbox("Select Country", unique_countries)

filtered_jobs = df[df["country"] == country_select][["title", "avg_hourly_rate", "country", "job_type"]]
if not filtered_jobs.empty:
    filtered_jobs["Formatted Salary"] = filtered_jobs.apply(lambda x: format_salary(x["avg_hourly_rate"], x["country"], "Mid-Level"), axis=1)
    st.dataframe(filtered_jobs.reset_index(drop=True))
else:
    st.warning("No jobs found for this country.")

# =========================
# Remote Jobs Section
# =========================
st.markdown('<h2 class="section-header">🏠 Remote Job Listings</h2>', unsafe_allow_html=True)
remote_jobs = df[df["title"].str.contains("remote", case=False, na=False)][["title", "avg_hourly_rate", "country", "job_type"]]
if not remote_jobs.empty:
    remote_jobs["Formatted Salary"] = remote_jobs.apply(lambda x: format_salary(x["avg_hourly_rate"], x["country"], "Mid-Level"), axis=1)
    st.dataframe(remote_jobs.reset_index(drop=True))
else:
    st.warning("No remote job listings found.")

# =========================
# Skill Gap Analyzer (Improved)
# =========================
st.markdown('<h2 class="section-header">🧠 Skill Gap Analyzer</h2>', unsafe_allow_html=True)
skills_input = st.text_input("Enter your current skills (comma-separated):")

if skills_input:
    tfidf_matrix = vectorizer.transform(df["title"].astype(str))
    query_vec = vectorizer.transform([skills_input])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = similarity.argsort()[-3:][::-1]
    matched_jobs = df.iloc[top_idx][["title", "job_type", "country"]]
    st.success("Here are some roles that best match your skills:")
    st.table(matched_jobs.reset_index(drop=True))
else:
    st.info("Enter your skills (e.g., Python, Excel, Machine Learning) to analyze skill fit and missing gaps.")

# =========================
# Resume Feedback Section
# =========================
st.markdown('<h2 class="section-header">📄 Resume Feedback Assistant</h2>', unsafe_allow_html=True)
experience_level = st.selectbox("Select Your Experience Level:", ["Fresher", "Mid-Level", "Senior"])
resume_text = st.text_area("Paste your resume text here:")

if st.button("Analyze Resume"):
    if resume_text.strip():
        tfidf_matrix = vectorizer.transform(df["title"].astype(str))
        query_vec = vectorizer.transform([resume_text])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_match_idx = similarity.argmax()
        best_job = df.iloc[best_match_idx]

        salary_display = format_salary(best_job["avg_hourly_rate"], best_job["country"], experience_level)
        job_type = best_job["job_type"]

        st.markdown("### 🧾 Resume Match Result")
        st.success(f"**🏢 Best Match:** {best_job['title']}")
        st.write(f"💰 **Expected Rate:** {salary_display}")
        st.write(f"🌍 **Country:** {best_job['country']}")
        st.write(f"🏷️ **Job Type:** {job_type}")

        st.markdown("### 💡 Personalized Feedback")
        st.info(f"Your profile aligns well with {best_job['title']}. Consider emphasizing key skills related to this role to improve job match quality.")
    else:
        st.warning("Please paste your resume text before analysis.")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("<center><b>🚀 Designed with ❤️ by Deepa Pathak | Internship Project</b></center>", unsafe_allow_html=True)
