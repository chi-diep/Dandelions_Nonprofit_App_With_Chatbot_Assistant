"""
NOTE: This is an alternative way to build the Volunteer RAG Assistant.

This loads all volunteer, shift, kit, signup, and story data directly from JSON files. It then builds a Chroma vector store for Retrieval-Augmented Generation (RAG)
using HuggingFace embeddings and answers questions with Ollama (Mistral).

This is ideal for:
- simpler demos that don't require a running database
- easy deployment of static data
- showing how JSON-based workflows can achieve similar results.

Make sure your JSON files are in the 'Data- JSON format/' folder.
"""

from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
import json
import os
import time

# 1) Load data from JSON
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

# 2) Build profiles
profiles = {}
for v in volunteers:
    vid = v.get("volunteer_id")
    profiles[vid] = {
        "profile": f"Volunteer Profile:\nID: {vid}\nName: {v.get('first_name')} {v.get('last_name')}\nDOB: {v.get('dob')}\nEmail: {v.get('email')}\nAddress: {v.get('address')}\nTitle: {v.get('title')}\nDays Available: {v.get('days_available')}",
        "kits": [],
        "shifts": [],
        "stories": []
    }

for k in kits:
    vid = k.get("volunteer_id")
    if vid in profiles:
        profiles[vid]["kits"].append(f"Kit ID: {k['kit_id']}, Type: {k['kit_type']}, Quantity: {k['quantity']}, Date: {k['date']}, Location: {k['location']}")

for s in shifts:
    vid = s.get("volunteer_id")
    if vid in profiles:
        profiles[vid]["shifts"].append(f"Shift ID: {s['shift_id']}, Title: {s['title']}, Hours: {s['hours']}, Date: {s['date']}")

for story in stories:
    vid = story.get("volunteer_id")
    if vid in profiles:
        profiles[vid]["stories"].append(f"Story ID: {story['story_id']}, Related Shift: {story['related_shift_id']}, Related Kit: {story['related_kit_id']}, Text: {story['text']}")

# turn signups into pseudo-profiles
for s in signups:
    sid = s.get("id")
    pseudo_id = f"signup_{sid}"
    profiles[pseudo_id] = {
        "profile": f"Signup:\nID: {sid}\nName: {s['name']}\nEmail: {s['email']}\nPhone: {s['phone']}\nShift: {s['shift_title']}, Date: {s['shift_date']}, Hours: {s['shift_hours']}\nSigned at: {s['created_at']}",
        "kits": [],
        "shifts": [],
        "stories": []
    }

st.success(f"Loaded {len(profiles)} profiles from JSON.")

# 3) Build documents for RAG
documents = []
for vid, data in profiles.items():
    total_hours = 0
    for s in data["shifts"]:
        parts = s.split("Hours:")
        if len(parts) > 1:
            try:
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

st.info(f"Created {len(documents)} documents for Chroma.")

# 4) Vector store & QA
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = splitter.split_documents(documents)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
db = Chroma.from_documents(chunks, embedding=embedder)
retriever = db.as_retriever(search_kwargs={"k": 10})

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

llm = Ollama(model="mistral")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# 5) Streamlit UI
st.title("Volunteer RAG Assistant (JSON + Ollama)")
st.write("Ask anything about the volunteers, like:")
st.markdown("""
- What is the ID of Shannon Hamilton?
- Who is available on Monday?
- What volunteers distributed kits and then worked more than 4 hours?
""")

query = st.text_input("Your question:")

if query:
    start_time = time.time()
    try:
        result = qa_chain.invoke({"query": query})
        elapsed = time.time() - start_time
        st.subheader("Answer:")
        st.write(result.get("result", "I don't know."))
        st.caption(f"Generated in {elapsed:.2f} seconds")
    except Exception as e:
        st.error(f"Failed to generate answer: {e}")
