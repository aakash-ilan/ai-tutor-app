import streamlit as st
import os
from pathlib import Path
from rag import ingest

upload_dir = Path("data/uploaded_books")
st.set_page_config(page_title="📘 AI Tutor - Upload Your Book")

uploaded_file = st.file_uploader("Choose a Book to upload (PDF file)", type=["pdf"])
if uploaded_file is not None:

    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

files = sorted([f for f in upload_dir.iterdir() if f.suffix == ".pdf"])
st.title("📚 Your Uploaded Books")
selected_files = []

if files:
    st.subheader("Select books to process:")
    for file in files:
        col1, col2 = st.columns([6, 1])
        with col1:
            if st.checkbox(file.name, key=str(file)):
                selected_files.append(file)
        with col2:
            if st.button("🗑️", key=f"delete_{file}"):
                file.unlink()  # Delete the file
                st.warning(f"Deleted: {file.name}")
                st.rerun()
else:
    st.markdown("No Books found")

if selected_files:
    st.success(f"Selected {len(selected_files)} file(s):")
    for f in selected_files:
        st.markdown(f"- ✅ `{f.name}`")
    file_list_with_path = list(map(lambda x: Path(x), selected_files))
if st.button("🚀 Process Selected Files"):
    message_placeholder = st.empty()
    message_placeholder.info("Processing started...")
    ingest(selected_files)
    message_placeholder.info("Completed.")

