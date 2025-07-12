# rag_chain.py
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import psycopg2

def build_rag_chain():
    profiles = {}
    conn = None
    try:
        # connect to my postgres database
        conn = psycopg2.connect(
        dbname="Dandelions",
        user="postgres",
        password=os.getenv("DB_PASSWORD"),
        host="localhost",
        port="5432"
        )

        cur = conn.cursor()

        # grab volunteers
        cur.execute("SELECT volunteer_id, first_name, last_name, dob, email, address, title, days_available FROM volunteers;")
        for row in cur.fetchall():
            vid, fname, lname, dob, email, address, title, days_avail = row
            profiles[vid] = {
                "profile": f"Volunteer Profile:\nID: {vid}\nName: {fname} {lname}\nDOB: {dob}\nEmail: {email}\nAddress: {address}\nTitle: {title}\nDays Available: {days_avail}",
                "kits": [], "shifts": [], "stories": [], "signups": []
            }

        # grab kits
        cur.execute("SELECT volunteer_id, kit_id, kit_type, quantity, date_given, location FROM kits;")
        for vid, kid, ktype, qty, date, loc in cur.fetchall():
            if vid in profiles:
                profiles[vid]["kits"].append(f"Kit ID: {kid}, Type: {ktype}, Quantity: {qty}, Date: {date}, Location: {loc}")

        # grab shifts
        cur.execute("SELECT volunteer_id, shift_id, title, hours, date FROM shifts;")
        for vid, sid, title, hours, date in cur.fetchall():
            if vid in profiles:
                profiles[vid]["shifts"].append(f"Shift ID: {sid}, Title: {title}, Hours: {hours}, Date: {date}")

        # grab stories
        cur.execute("SELECT volunteer_id, story_id, text, related_shift_id, related_kit_id FROM personal_stories;")
        for vid, sid, text, rel_shift, rel_kit in cur.fetchall():
            if vid in profiles:
                profiles[vid]["stories"].append(f"Story ID: {sid}, Related Shift: {rel_shift}, Related Kit: {rel_kit}, Text: {text}")

        # grab signups
        cur.execute("SELECT id, name, email, phone, shift_title, shift_date, shift_hours, created_at FROM signups;")
        for sid, name, email, phone, stitle, sdate, shours, created in cur.fetchall():
            pseudo_id = f"signup_{sid}"
            profiles[pseudo_id] = {
                "profile": f"Signup:\nID: {sid}\nName: {name}\nEmail: {email}\nPhone: {phone}\nShift: {stitle}, Date: {sdate}, Hours: {shours}\nSigned at: {created}",
                "kits": [], "shifts": [], "stories": [], "signups": []
            }

        cur.close()
        print(f"Loaded data: {len(profiles)} volunteer & signup profiles.")
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None, None

    # turn volunteer data into documents for RAG
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

    print(f"Created {len(documents)} documents for RAG.")

    # set up vector database with chroma
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    db = Chroma.from_documents(chunks, embedding=embedder)
    retriever = db.as_retriever(search_kwargs={"k": 10})
    print(f"Chroma vector DB ready with {len(chunks)} chunks.")

    # build the QA chain that does the answering
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant managing volunteer data.

Use ONLY the context below.
If the question involves shifts or kits, make sure to consider total hours and kit distribution.
If the info is not there, reply: "I don't know."

Context:
{context}

Question: {question}

Answer:"""
    )
    llm = OllamaLLM(model="mistral")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

    print("RAG + hybrid chain fully initialized.")
    return qa_chain, conn

# this handles questions that might be better answered by SQL
def hybrid_qa(query, qa_chain, conn):
    lower_q = query.lower()
    try:
        if any(word in lower_q for word in ["count", "total", "number of", "how many", "sum", "max", "min", "largest", "smallest", "average"]):
            with conn.cursor() as cur:
                if "volunteer" in lower_q:
                    print("Running SQL COUNT on volunteers...")
                    cur.execute("SELECT COUNT(*) FROM volunteers;")
                    count = cur.fetchone()[0]
                    return f"There are a total of {count} volunteers in your dataset."
                elif "kits" in lower_q:
                    print("Running SQL COUNT on kits...")
                    cur.execute("SELECT COUNT(*) FROM kits;")
                    count = cur.fetchone()[0]
                    return f"There are a total of {count} kits distributed."
                elif "shifts" in lower_q:
                    print("Running SQL COUNT on shifts...")
                    cur.execute("SELECT COUNT(*) FROM shifts;")
                    count = cur.fetchone()[0]
                    return f"There are a total of {count} shifts recorded."
                else:
                    return "I can compute totals and counts, but please specify 'volunteers', 'kits', or 'shifts'."
    except Exception as e:
        print(f"SQL failed: {e}")

    print("Falling back to RAG...")
    rag_result = qa_chain.invoke({"query": query})
    return rag_result.get("result", "I don't know.")
