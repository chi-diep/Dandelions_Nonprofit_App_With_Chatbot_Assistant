from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import psycopg2
import streamlit as st
import time

# 1) Getting all the data from my Postgres database
profiles = {}
try:
    conn = psycopg2.connect(
        dbname="Dandelions",
        user="postgres",
        password=os.getenv("DB_PASSWORD"),
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    # grabbing volunteer details
    cur.execute("SELECT volunteer_id, first_name, last_name, dob, email, address, title, days_available FROM volunteers;")
    for row in cur.fetchall():
        vid, fname, lname, dob, email, address, title, days_avail = row
        profiles[vid] = {
            "profile": f"Volunteer Profile:\nID: {vid}\nName: {fname} {lname}\nDOB: {dob}\nEmail: {email}\nAddress: {address}\nTitle: {title}\nDays Available: {days_avail}",
            "kits": [],
            "shifts": [],
            "stories": [],
            "signups": []
        }

    # grabbing kits info
    cur.execute("SELECT volunteer_id, kit_id, kit_type, quantity, date_given, location FROM kits;")
    for vid, kid, ktype, qty, date, loc in cur.fetchall():
        if vid in profiles:
            profiles[vid]["kits"].append(f"Kit ID: {kid}, Type: {ktype}, Quantity: {qty}, Date: {date}, Location: {loc}")

    # grabbing shift info
    cur.execute("SELECT volunteer_id, shift_id, title, hours, date FROM shifts;")
    for vid, sid, title, hours, date in cur.fetchall():
        if vid in profiles:
            profiles[vid]["shifts"].append(f"Shift ID: {sid}, Title: {title}, Hours: {hours}, Date: {date}")

    # grabbing stories
    cur.execute("SELECT volunteer_id, story_id, text, related_shift_id, related_kit_id FROM personal_stories;")
    for vid, sid, text, rel_shift, rel_kit in cur.fetchall():
        if vid in profiles:
            profiles[vid]["stories"].append(f"Story ID: {sid}, Related Shift: {rel_shift}, Related Kit: {rel_kit}, Text: {text}")

    # grabbing signups
    cur.execute("SELECT id, name, email, phone, shift_title, shift_date, shift_hours, created_at FROM signups;")
    for sid, name, email, phone, stitle, sdate, shours, created in cur.fetchall():
        pseudo_id = f"signup_{sid}"
        profiles[pseudo_id] = {
            "profile": f"Signup:\nID: {sid}\nName: {name}\nEmail: {email}\nPhone: {phone}\nShift: {stitle}, Date: {sdate}, Hours: {shours}\nSigned at: {created}",
            "kits": [],
            "shifts": [],
            "stories": [],
            "signups": []
        }

    cur.close()
    conn.close()
except Exception as e:
    st.error(f"Database connection failed: {e}")

# 2) Making all the volunteer documents for RAG
documents = []
for vid, data in profiles.items():
    if data["kits"] or data["shifts"] or data["stories"] or data["signups"]:
        total_hours = 0
        for s in data["shifts"]:
            parts = s.split("Hours:")
            if len(parts) > 1:
                try:
                    total_hours += float(parts[1].split(",")[0].strip())
                except:
                    pass
        content = f"{data['profile']}\n"
        content += f"Total hours worked across shifts: {total_hours}\n\n"

        if data["kits"]:
            content += "Kits Given:\n" + "\n".join(data["kits"]) + "\n"
        if data["shifts"]:
            content += "Shifts Worked:\n" + "\n".join(data["shifts"]) + "\n"
        if data["stories"]:
            content += "Personal Stories:\n" + "\n".join(data["stories"]) + "\n"

        documents.append(Document(page_content=content))

print(f"Created {len(documents)} volunteer + multi-hop documents.")

# 3) Making the Chroma vector database
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
texts = splitter.split_documents(documents)
texts = [doc for doc in texts if doc.page_content.strip()]
print(f"Split into {len(texts)} text chunks.")

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
db = Chroma.from_documents(texts, embedding=embedder)
print("Chroma vector DB created.")

retriever = db.as_retriever(search_kwargs={"k": 10})

# 4) Making the QA chain that does the answering
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
print("QA chain ready.")

# 5) Streamlit app for asking stuff
st.title("Volunteer RAG Assistant (Fast Mistral)")
st.write("Ask anything about the volunteers, like:")
st.markdown("""
- What is the ID of Shannon Hamilton?
- Who is available on Monday?
- What volunteers distributed kits and then worked more than 4 hours?
""")

query = st.text_input("Your question:")

if query:
    try:
        start_time = time.time()
        answer = qa_chain.invoke({"query": query})
        elapsed = time.time() - start_time
        st.write(f"**Answer:** {answer.get('result', 'No answer found')}")
        st.info(f"Answer generated in {elapsed:.2f} seconds")
    except Exception as e:
        st.error(f"Failed to generate an answer: {e}")
