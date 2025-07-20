import streamlit as st
import json
import os
import time
from threading import Thread
from flask import Flask, request, jsonify, send_from_directory
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load each JSON file from the data folder
def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Failed loading {path}: {e}")
        return []

volunteers = load_json("volunteers.json")
kits = load_json("kits.json")
shifts = load_json("shifts.json")
stories = load_json("personal_stories.json")
signups = load_json("signups_data.json")

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

for s in signups:
    sid = s.get("id")
    pseudo_id = f"signup_{sid}"
    profiles[pseudo_id] = {
        "profile": f"Signup:\nID: {sid}\nName: {s['name']}\nEmail: {s['email']}\nPhone: {s['phone']}\nShift: {s['shift_title']}, Date: {s['shift_date']}, Hours: {s['shift_hours']}\nSigned at: {s['created_at']}",
        "kits": [],
        "shifts": [],
        "stories": []
    }

for k in kits:
    vid = k.get("volunteer_id")
    if vid in profiles:
        profiles[vid]["kits"].append(
            f"Kit ID: {k['kit_id']}, Type: {k['kit_type']}, Quantity: {k['quantity']}, Date: {k['date_given']}, Location: {k['location']}")

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

# Create a document for each profile
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

# Split text for better retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Embed and store in Chroma
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
db = Chroma.from_documents(chunks, embedding=embedder)
retriever = db.as_retriever(search_kwargs={"k": 10})

# Build the prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use ONLY the context below to answer.
If the info is not there, say: I don't know.

Context:
{context}

Question: {question}
Answer:"""
)

# Manual RAG chain
llm = Ollama(model="tinyllama")
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    input_key="query"
)

# Detect if the question is statistical
stat_keywords = ["how many", "count", "total", "sum", "average", "mean", "min", "max"]
def is_stat_question(q):
    return any(k in q.lower() for k in stat_keywords)

def answer_with_stats(q):
    if "volunteers" in q.lower():
        return f"There are {len(volunteers)} volunteers."
    if "kits" in q.lower():
        return f"Total kits given out: {len(kits)}"
    if "shifts" in q.lower():
        return f"Total shifts: {len(shifts)}"
    if "stories" in q.lower():
        return f"Number of personal stories: {len(stories)}"
    return "I don't know."

def hybrid_qa(question):
    if is_stat_question(question):
        return answer_with_stats(question)
    else:
        result = rag_chain.invoke({"query": question})
        return result.get("result", "I don't know.")

# Streamlit UI
st.set_page_config(page_title="Dandelions RAG Assistant", page_icon="🌼")
st.title("Volunteer RAG Assistant")
st.caption("Ask about volunteers, kits, shifts or stories")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Clear Chat"):
    st.session_state.chat_history = []

for q, a in st.session_state.chat_history[::-1]:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Dandy:** {a}")

query = st.text_input("Ask a question")
if query:
    with st.spinner("Thinking..."):
        start = time.time()
        try:
            answer = hybrid_qa(query)
            st.session_state.chat_history.append((query, answer))
            st.markdown(f"**Dandy:** {answer}")
            st.caption(f"Generated in {time.time() - start:.2f}s")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
