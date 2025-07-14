# rag_chain.py
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import json
import os

def build_rag_chain():
    profiles = {}

    try:
        # load data from JSON files
        with open('Data- JSON format/volunteers.json', 'r', encoding='utf-8') as f:
            volunteers = json.load(f)
        with open('Data- JSON format/kits.json', 'r', encoding='utf-8') as f:
            kits = json.load(f)
        with open('Data- JSON format/shifts.json', 'r', encoding='utf-8') as f:
            shifts = json.load(f)
        with open('Data- JSON format/personal_stories.json', 'r', encoding='utf-8') as f:
            stories = json.load(f)
        with open('Data- JSON format/signups_data.json', 'r', encoding='utf-8') as f:
            signups = json.load(f)

        # build profiles from volunteers
        for v in volunteers:
            vid = v['volunteer_id']
            profiles[vid] = {
                "profile": f"Volunteer Profile:\nID: {vid}\nName: {v['first_name']} {v['last_name']}\nDOB: {v['dob']}\nEmail: {v['email']}\nAddress: {v['address']}\nTitle: {v['title']}\nDays Available: {v['days_available']}",
                "kits": [], "shifts": [], "stories": [], "signups": []
            }

        # kits
        for k in kits:
            vid = k['volunteer_id']
            if vid in profiles:
                profiles[vid]["kits"].append(
                    f"Kit ID: {k['kit_id']}, Type: {k['kit_type']}, Quantity: {k['quantity']}, Date: {k['date']}, Location: {k['location']}"
                )

        # shifts
        for s in shifts:
            vid = s['volunteer_id']
            if vid in profiles:
                profiles[vid]["shifts"].append(
                    f"Shift ID: {s['shift_id']}, Title: {s['title']}, Hours: {s['hours']}, Date: {s['date']}"
                )

        # stories
        for story in stories:
            vid = story['volunteer_id']
            if vid in profiles:
                profiles[vid]["stories"].append(
                    f"Story ID: {story['story_id']}, Related Shift: {story['related_shift_id']}, Related Kit: {story['related_kit_id']}, Text: {story['text']}"
                )

        # also create profiles for signups
        for signup in signups:
            pseudo_id = f"signup_{signup['id']}"
            profiles[pseudo_id] = {
                "profile": f"Signup:\nID: {signup['id']}\nName: {signup['name']}\nEmail: {signup['email']}\nPhone: {signup['phone']}\nShift: {signup['shift_title']}, Date: {signup['shift_date']}, Hours: {signup['shift_hours']}\nSigned at: {signup['created_at']}",
                "kits": [], "shifts": [], "stories": [], "signups": []
            }

        print(f"Loaded data: {len(profiles)} volunteer & signup profiles.")

    except Exception as e:
        print(f"Failed to load JSON data: {e}")
        return None

    # turn profiles into documents
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

    # setup vector db
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    db = Chroma.from_documents(chunks, embedding=embedder)
    retriever = db.as_retriever(search_kwargs={"k": 10})

    # build the QA chain
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

    print("RAG chain fully initialized.")
    return qa_chain


def hybrid_qa(query, qa_chain):
    lower_q = query.lower()
    try:
        if any(word in lower_q for word in ["count", "total", "number of", "how many", "sum", "max", "min", "largest", "smallest", "average"]):
            if "volunteer" in lower_q:
                with open('Data- JSON format/volunteers.json', 'r', encoding='utf-8') as f:
                    volunteers = json.load(f)
                return f"There are a total of {len(volunteers)} volunteers in your dataset."
            elif "kits" in lower_q:
                with open('Data- JSON format/kits.json', 'r', encoding='utf-8') as f:
                    kits = json.load(f)
                return f"There are a total of {len(kits)} kits distributed."
            elif "shifts" in lower_q:
                with open('Data- JSON format/shifts.json', 'r', encoding='utf-8') as f:
                    shifts = json.load(f)
                return f"There are a total of {len(shifts)} shifts recorded."
            else:
                return "I can compute totals and counts, but please specify 'volunteers', 'kits', or 'shifts'."
    except Exception as e:
        print(f"JSON lookup failed: {e}")

    print("Falling back to RAG...")
    rag_result = qa_chain.invoke({"query": query})
    return rag_result.get("result", "I don't know.")
