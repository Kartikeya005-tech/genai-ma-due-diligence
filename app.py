import streamlit as st
import urllib.request
import json
import time
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# --- PAGE CONFIGURATION & CORPORATE STYLING ---
st.set_page_config(page_title="M&A Due Diligence Engine", page_icon="💼", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #FAFAFA; }
    .css-1d391kg { background-color: #005A36; }
    h1, h2, h3 { color: #005A36; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { background-color: #005A36; color: white; border-radius: 4px; border: none; width: 100%; }
    .stButton>button:hover { background-color: #003d24; }
    .status-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    /* Forces all paragraphs and bullet points to be readable */
    .stAlert p, .stAlert li { color: #111111 !important; }
    div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li { color: #111111 !important; }
</style>
""", unsafe_allow_html=True)

# --- DIAGNOSTIC FUNCTIONS ---
import urllib.request
import urllib.error
import json

def validate_api_key(api_key):
    """Pings the Gemini API directly with a strict 5-second timeout."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        data = json.dumps({"contents": [{"parts": [{"text": "ping"}]}]}).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        # The 5-second cutoff prevents infinite spinning
        response = urllib.request.urlopen(req, timeout=5)
        return True, "API Key is valid and active."
        
    except urllib.error.HTTPError as e:
        if e.code == 400: return False, "Invalid API Key format."
        if e.code == 403: return False, "API Key is not authorized."
        return False, f"Google rejected the key: HTTP {e.code}"
    except Exception as e:
        return False, f"Network blocked the request. Try a Mobile Hotspot. ({str(e)})"

# --- CORE RAG FUNCTIONS ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Change your chunking function to look exactly like this:
def get_text_chunks(text):
    # Increased chunk size to 10,000 to bypass the free-tier 100-request limit
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        
        # 1. Initialize the database with just the very first chunk
        vector_store = FAISS.from_texts([text_chunks[0]], embedding=embeddings)
        
        # 2. Loop through the remaining chunks one single piece at a time
        for i in range(1, len(text_chunks)):
            vector_store.add_texts([text_chunks[i]])
            time.sleep(5)  # 5-second pause = exactly 12 requests per minute (well under the limit)
            
        vector_store.save_local("faiss_index")
        return True, "Vector database successfully built and saved."
    except Exception as e:
        return False, f"Embedding failed: {str(e)}"

# Add api_key as a parameter in the parentheses
def get_conversational_chain(api_key):
    prompt_template = """
    You are an expert M&A consultant conducting due diligence. Answer the question based strictly on the provided context from the corporate data room documents. 
    If the answer is not in the provided context, state: "This information is not present in the current data room files." Do not invent information.
    Provide your answer in a clear, highly professional, executive-summary format. Use bullet points for readability.

    Context: \n {context}?\n
    Question: \n {question}\n

    Answer:
    """
    # Add google_api_key=api_key to the model definition
    # Change this line:
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Change this line to pass the api_key
    chain = get_conversational_chain(api_key)
    
    # Change your response line to this:
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.markdown("### 📄 Executive Summary")
    st.info(response["output_text"])

# --- USER INTERFACE ---
def main():
    st.title("💼 GenAI M&A Due Diligence Engine")
    st.markdown("Upload target company documentation (financials, contracts, disclosures) to identify risks and extract key clauses instantly.")

    with st.sidebar:
        st.header("1. Authentication")
        api_key = st.text_input("Enter Google Gemini API Key:", type="password")
        
        # Live Validation Button
        if st.button("Validate Key"):
            if not api_key:
                st.warning("Please enter a key first.")
            else:
                with st.spinner("Pinging Google Servers..."):
                    is_valid, msg = validate_api_key(api_key)
                    if is_valid:
                        st.success("✅ Key is active! You may proceed.")
                    else:
                        st.error(f"❌ Key Error: {msg}")
        
        st.markdown("[Get a free Gemini API key here](https://aistudio.google.com/app/apikey)")
        st.divider()
        
        st.header("2. Data Room Upload")
        pdf_docs = st.file_uploader("Upload PDF Documents", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if not api_key:
                st.error("Please enter your API Key in step 1.")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                # Step-by-step UI Status Updates
                status_text = st.empty()
                progress_bar = st.progress(0)
                
                status_text.info("Step 1/3: Extracting text from PDFs...")
                raw_text = get_pdf_text(pdf_docs)
                progress_bar.progress(33)
                
                status_text.info("Step 2/3: Chunking text for analysis...")
                text_chunks = get_text_chunks(raw_text)
                progress_bar.progress(66)
                
                status_text.info("Step 3/3: Generating AI Embeddings (This may take a minute)...")
                success, msg = get_vector_store(text_chunks, api_key)
                
                if success:
                    progress_bar.progress(100)
                    status_text.success(f"✅ {msg}")
                    st.balloons() # Visual confirmation for the user
                else:
                    progress_bar.progress(0)
                    status_text.error(f"❌ {msg}")

    # Main Chat Interface
    st.subheader("3. Query the Data Room")
    user_question = st.text_input("Ask a question (e.g., 'What are the change-of-control clauses in these contracts?')")

    if user_question:
        if not api_key:
            st.error("Please provide an API Key in the sidebar.")
        elif not os.path.exists("faiss_index"):
            st.warning("Please upload and process documents first. The vector database is missing.")
        else:
            with st.spinner("Analyzing data room..."):
                try:
                    user_input(user_question, api_key)
                except Exception as e:
                    st.error(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()