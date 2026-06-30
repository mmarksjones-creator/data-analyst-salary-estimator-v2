import streamlit as st
import pandas as pd
import joblib

model = joblib.load("salary_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Data Analyst Salary Estimator")
st.write("Estimate a salary based on experience, education, skills, and job context — trained on a 250,000-record dataset using a Random Forest model.")

job_title = st.selectbox("Job Title", ["ai engineer", "data analyst", "frontend developer", "business analyst",
                                         "product manager", "backend developer", "machine learning engineer",
                                         "devops engineer", "software engineer", "cybersecurity analyst",
                                         "data scientist", "cloud engineer"])
experience_years = st.slider("Years of Experience", 0, 25, 2)
education_level = st.selectbox("Education Level", ["high school", "diploma", "bachelor", "master", "phd"])
skills_count = st.slider("Number of Skills", 0, 30, 10)
industry = st.selectbox("Industry", ["healthcare", "telecom", "media", "retail", "manufacturing",
                                       "education", "finance", "technology", "consulting", "government"])
company_size = st.selectbox("Company Size", ["small", "medium", "large", "enterprise", "startup"])
location = st.selectbox("Location", ["india", "australia", "singapore", "canada", "sweden", "usa", "netherlands", "remote", "germany", "uk"])
remote_work = st.selectbox("Remote Work", ["yes", "no", "hybrid"])
certifications = st.slider("Number of Certifications", 0, 10, 1)

if st.button("Estimate Salary"):
    input_df = pd.DataFrame([{
        "job_title": job_title,
        "experience_years": experience_years,
        "education_level": education_level,
        "skills_count": skills_count,
        "industry": industry,
        "company_size": company_size,
        "location": location,
        "remote_work": remote_work,
        "certifications": certifications
    }])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    st.success(f"Estimated Salary: ${prediction:,.0f}")