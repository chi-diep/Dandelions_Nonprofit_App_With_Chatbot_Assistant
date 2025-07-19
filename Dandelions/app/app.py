import streamlit as st
import json
import os
import time
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load each JSON file from the data folder
def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed loading {path}: {e}")
        return []

volunteers = load_json("Data- JSON format/volunteers.json")
kits = load_json("Data- JSON format/kits.json")
shifts = load_json("Data- JSON format/shifts.json")
stories = load_json("Data- JSON format/personal_stories.json")
signups = load_json("Data- JSON format/signups_data.json")

# Build a profile for each volunteer or signup
profiles = {}
for v in volunteers:
    vid = v.get("volunteer_id")
    profiles[vid] = {
        "profile": f"Volunteer Profile:\nID: {vid}\nName: {v.get('first_name')} {v.get('last_name')}\nDOB: {v.get('dob')}\nEmail: {v.get('email')}\nAddress: {v.get('address')}\nTitle: {v.get('title')}\nDays Available: {v.get('days_available')}",
        "kits": [],
        "shifts": [],
        "stories": []
    }

# Add kits, shifts, and stories to each profile
for k in kits:
    vid = k.get("volunteer_id")
    if vid in profiles:
        profiles[vid]["kits"].append(
            f"Kit ID: {k['kit_id']}, Type: {k['kit_type']}, Quantity: {k['quantity']}, Date: {k['date']}, Location: {k['location']}")

for s in shifts:
    vid = s.get("volunteer_id")
    if vid in profiles:
        profiles[vid]["shifts"].append(
            f"Shift ID: {s['shift_id']}, Title: {s['title']}, Hours: {s['hours']}, Date: {s['date']}")

for story in stories:
    vid = story.get("volunteer_id")
    if vid in profiles:
        profiles[vid]["stories"].append(
            f"Story ID: {story['story_id']}, Related Shift: {story['related_shift_id']}, Related Kit: {story['related_kit_id']}, Text: {story['text']}")

# Turn each signup into a temporary profile too
for s in signups:
    sid = s.get("id")
    pseudo_id = f"signup_{sid}"
    profiles[pseudo_id] = {
        "profile": f"Signup:\nID: {sid}\nName: {s['name']}\nEmail: {s['email']}\nPhone: {s['phone']}\nShift: {s['shift_title']}, Date: {s['shift_date']}, Hours: {s['shift_hours']}\nSigned at: {s['created_at']}",
        "kits": [],
        "shifts": [],
        "stories": []
    }

# Create a document for each profile and include kits, shifts, stories
documents = []
for vid, data in profiles.items():
    total_hours = 0
    for s in data["shifts"]:
        try:
            parts = s.split("Hours:")
            if len(parts) > 1:
                total_hours += float(parts[1].split(",")[0].strip())
        except:
            pass

    content = f"{data['profile']}\nTotal hours worked across shifts: {total_hours}\n\n"
    if data["kits"]:
        content += "Kits Given:\n" + "\n".join(data["kits"]) + "\n"
    if data["shifts"]:
        content += "Shifts Worked:\n" + "\n".join(data["shifts"]) + "\n"
    if data["stories"]:
        content += "Personal Stories:\n" + "\n".join(data["stories"]) + "\n"

    documents.append(Document(page_content=content))

# Break documents into smaller chunks for better retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Create embeddings and load them into ChromaDB
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
db = Chroma.from_documents(chunks, embedding=embedder)
retriever = db.as_retriever(search_kwargs={"k": 10})

# Prompt to guide how the model should answer
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant managing volunteer data.

Use ONLY the context below. 
If the question involves shifts or kits, make sure to consider total hours and kit distribution.
If the info is not there, reply: "I don't know."

Context:
{context}

Question: {question}

Answer:"""
)

# Load Ollama model (like Mistral) and build the QA chain
llm = Ollama(model="mistral")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# Set up the Streamlit page
st.set_page_config(page_title="Dandelions RAG Assistant", page_icon="🌼")
st.title("🌼 Volunteer RAG Assistant")
st.caption("Ask anything about volunteers, shifts, kits, or their stories.")

# Start chat history if not already started
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Button to clear the chat
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# Show chat history from newest to oldest
for q, a in st.session_state.chat_history[::-1]:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Dandy:** {a}")

# Input box for user question
query = st.text_input("Your question:")

# Optional helper to detect stats questions
def is_statistical(q):
    keywords = ["how many", "count", "total", "sum", "average", "mean", "min", "max"]
    return any(k in q.lower() for k in keywords)

# Run the QA chain when a question is submitted
if query:
    with st.spinner("Thinking..."):
        start_time = time.time()
        try:
            result = qa_chain.invoke({"question": query})
            elapsed = time.time() - start_time
            answer = result.get("result", "I don't know.")
            st.session_state.chat_history.append((query, answer))
            st.markdown(f"**Dandy:** {answer}")
            st.caption(f"Generated in {elapsed:.2f} seconds")
        except Exception as e:
            st.error(f" Error: {e}")
