import streamlit as st
from analyzer_chain import extract_job, summarize_resume

st.set_page_config(page_title="AI Job Analyzer", layout="wide")
st.title("📝 AI-Powered Job Description Analyzer")

uploaded_file = st.file_uploader("Upload a Job Description document (.txt)", type=["txt", "pdf"])

if uploaded_file:
    job_text = uploaded_file.read()
    st.subheader("📄 Uploaded Resume Text")
    st.text_area("Resume", job_text, height=300)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing Job Description..."):
            job_result = extract_job(job_text)
            summary = summarize_resume(job_text)
            
            st.success("✅ Analysis complete!")

            st.subheader("Resume Info")
            st.markdown(f"** Name:** {job_result.name}")
            st.markdown(f"** Experience:** {job_result.total_experience}")
            st.markdown(f"** Skills:** {', '.join(job_result.skills)}")
            st.markdown(f"** Education:** {job_result.education}")

        st.subheader("📝 Resume Summary")
        st.write(summary.content)  # Display the content of the summary