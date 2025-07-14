import streamlit as st
from legal_chain import extract_contract_clauses, summarize_contract

st.set_page_config(page_title="AI Legal Analyzer", layout="wide")
st.title("📜 AI-Powered Legal Document Analyzer")

uploaded_file = st.file_uploader("Upload a legal document (.txt)", type=["txt"])

if uploaded_file:
    contract_text = uploaded_file.read().decode("utf-8")
    
    st.subheader("📄 Uploaded Contract Text")
    st.text_area("Contract", contract_text, height=300)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing contract..."):
            clauses_result = extract_contract_clauses(contract_text)
            summary = summarize_contract(contract_text)

        if clauses_result:
            st.subheader("📚 Extracted Clauses")
            for clause in clauses_result.clauses:
                st.markdown(f"### {clause.title}")
                st.write(clause.content)
        else:
            st.error("❌ Clause extraction failed. Please check the input contract.")

        st.subheader("📝 Contract Summary")
        st.success(summary)
