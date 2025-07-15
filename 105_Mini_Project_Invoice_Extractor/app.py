import streamlit as st
from invoice_chain import extract_invoice

st.set_page_config(page_title="AI Invoice Analyzer", layout="wide")
st.title("📜 AI-Powered Invoice Document Analyzer")

uploaded_file = st.file_uploader("Upload a Invoice document (.txt)", type=["txt"])

if uploaded_file:
    invoice_text = uploaded_file.read().decode("utf-8")
    
    st.subheader("📄 Uploaded Invoice Text")
    st.text_area("Invoice", invoice_text, height=300)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing Invoice..."):
            invoice_result = extract_invoice(invoice_text)
            
            st.success("✅ Analysis complete!")

            st.subheader("Invoice Info")
            st.markdown(f"**Invoice Number:** {invoice_result.invoice_number}")
            st.markdown(f"**Date:** {invoice_result.invoice_date}")

            st.markdown(f"**Vendor:** {invoice_result.vendor}")
            st.markdown("### Line Items")
            items_data = [
                {
                    "Description": item.description,
                    "Quantity": item.quantity,
                    "Unit Price": item.unit_price,
                    "Total": item.total
                }
                for item in invoice_result.line_items
            ]
            st.table(items_data)

            st.markdown(f"**Total Amount:** {invoice_result.total_amount}")

# This item_tabel will display line item with indexes from 0 onwards in order to process that wihout line index either user Pandas or custom HTML table       
