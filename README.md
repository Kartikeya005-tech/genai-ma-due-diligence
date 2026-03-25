# 💼 GenAI M&A Due Diligence Engine

**Live Application:** [Insert Your Streamlit Cloud URL Here]

## 📌 Executive Summary
During the Mergers & Acquisitions (M&A) due diligence process, consulting teams and analysts spend hundreds of hours manually reviewing data rooms containing dense corporate filings, financial disclosures, and complex legal contracts. 

This project is an enterprise-grade Retrieval-Augmented Generation (RAG) pipeline designed to automate that workflow. It ingests complex target company documentation and allows users to instantly query the data room to extract hard financial metrics, map conditional liability clauses, and synthesize regulatory risks.

## ⚙️ Technical Architecture
Built with a focus on mitigating API rate limits and optimizing context windows, this engine utilizes:
* **LLM Orchestration:** LangChain
* **Vector Database:** FAISS (Facebook AI Similarity Search) for rapid semantic retrieval
* **Embeddings:** Google `gemini-embedding-001` (Batched processing to handle token limits)
* **Generation Model:** Google `gemini-2.5-flash` for high-speed, structured text synthesis
* **Frontend UI:** Streamlit 

## 🚀 Key Features
* **Intelligent Chunking:** Automatically processes massive PDFs (like 80+ page 10-K reports) using `RecursiveCharacterTextSplitter` to maintain semantic integrity across document breaks.
* **Executive Formatting:** Prompt-engineered to respond strictly in the persona of an expert M&A consultant, ensuring all outputs are concise, factual, and heavily bulleted.
* **Hallucination Mitigation:** Grounded strictly in the provided data room context. If a query falls outside the uploaded documents, the system is hardcoded to state the information is missing rather than inventing an answer.

## 💻 Local Installation
To run this application on your local machine:
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Google API Key
4. Run the app: `streamlit run app.py`
