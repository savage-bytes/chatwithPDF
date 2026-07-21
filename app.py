import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA


# --- Load environment variables ---
load_dotenv()
UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- OpenAI API Key ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("🚨 OPENAI_API_KEY is missing! Please add it in your .env file.")


# --- Flask app config ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# --- OpenAI Models ---
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # ✅ Chat model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")  # ✅ Embeddings


# --- Global retriever (for demo purposes) ---
retriever = None


# --- Helper: process uploaded PDF ---
def process_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Split into chunks for retrieval
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(docs)

        # Create FAISS vector store
        vector_store = FAISS.from_documents(documents, embedding_model)
        return vector_store.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None


# --- Route: Upload PDF ---
@app.route("/", methods=["GET", "POST"])
def index():
    global retriever
    if request.method == "POST":
        uploaded_file = request.files.get("pdf_file")
        if uploaded_file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
            uploaded_file.save(file_path)
            retriever = process_pdf(file_path)
            if retriever:
                return redirect(url_for('chat'))
            else:
                return "Error processing PDF.", 500
    return render_template("index.html")


# --- Route: Chat with RAG ---
@app.route("/chat", methods=["GET", "POST"])
def chat():
    global retriever
    if request.method == "POST":
        data = request.get_json()
        user_input = data.get("message")
        history_json = data.get("history", [])

        if not user_input:
            return jsonify({"error": "No message provided."}), 400
        if not retriever:
            return jsonify({"error": "No document uploaded. Upload a PDF first."}), 400

        # Rebuild chat history
        chat_history = []
        for msg in history_json:
            if msg.get("type") == "human":
                chat_history.append(HumanMessage(content=msg.get("content")))
            elif msg.get("type") == "ai":
                chat_history.append(AIMessage(content=msg.get("content")))

        # Retrieval-Augmented Generation (RAG) chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Run query
        result = rag_chain({"query": user_input})
        answer = result["result"]
        sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]

        return jsonify({"response": answer, "sources": sources})

    return render_template("chat.html")


# --- Run app ---
if __name__ == "__main__":
    app.run(debug=True)
