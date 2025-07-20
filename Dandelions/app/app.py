import streamlit as st
import json
import os
import time
import re
from threading import Thread
from flask import Flask, request, jsonify, send_from_directory
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

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
        "kits": [], "shifts": [], "stories": []
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

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = splitter.split_documents(documents)

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
db = Chroma.from_documents(chunks, embedding=embedder)
retriever = db.as_retriever(search_kwargs={"k": 10})

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. ONLY use the context below to answer the question.
If the answer is not present in the context, say "I don't know."
NEVER guess or make up information.

Context:
{context}

Question: {question}
Answer:"""
)

llm = Ollama(model="tinyllama")
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    input_key="query"
)

stat_keywords = ["how many", "count", "total", "sum", "average", "mean", "min", "max"]
def is_stat_question(q):
    return any(k in q.lower() for k in stat_keywords)

def answer_with_stats(q):
    if "volunteers" in q: return f"There are {len(volunteers)} volunteers."
    if "kits" in q: return f"Total kits given out: {len(kits)}"
    if "shifts" in q: return f"Total shifts: {len(shifts)}"
    if "stories" in q: return f"Number of personal stories: {len(stories)}"
    return "I don't know."

def extract_name_and_field(question):
    question = question.lower()
    name_match = re.search(r"\b([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\b", question)
    field_keywords = {
        "volunteer_id": ["volunteer id", "id"],
        "dob": ["dob", "birthdate"],
        "days_available": ["availability", "days available"],
        "title": ["title", "role"],
        "email": ["email"],
        "address": ["address", "location"],
        "phone": ["phone"],
        "shift_title": ["shift title"],
        "shift_date": ["shift date"],
        "shift_hours": ["shift hours"],
        "created_at": ["signed at"]
    }
    field_match = None
    for field, keywords in field_keywords.items():
        if any(k in question for k in keywords):
            field_match = field
            break
    return name_match.group(0).strip().lower() if name_match else None, field_match

def find_matching(data_list, name, name_keys=("first_name", "last_name")):
    name = name.lower()
    matches = []
    for item in data_list:
        full_name = " ".join(item.get(k, "") for k in name_keys).strip().lower()
        if name in full_name:
            matches.append(item)
    return matches

def hybrid_qa(question):
    q = question.lower().strip()
    if is_stat_question(q):
        return answer_with_stats(q)

    name, field = extract_name_and_field(q)
    if name and field:
        for dataset, name_keys, label in [
            (volunteers, ("first_name", "last_name"), "Volunteer"),
            (signups, ("name",), "Signup"),
            (kits, ("volunteer_id",), "Kit"),
            (shifts, ("volunteer_id",), "Shift")
        ]:
            matches = find_matching(dataset, name, name_keys)
            if matches:
                return "\n".join(
                    f"[{label}] {m.get('name', m.get('volunteer_id', ''))} — {field.replace('_',' ').title()}: {m.get(field, 'Not available')}"
                    for m in matches
                )
        return f"Sorry, no matching information found for '{name.title()}' and field '{field}'."

    try:
        result = rag_chain.invoke({"query": question})
        return result.get("result", "I don't know.")
    except Exception as e:
        return f"Something went wrong with RAG: {e}"

st.set_page_config(page_title="Dandelions RAG Assistant")
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
