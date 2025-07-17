# rag_chain.py

import os
import json
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

JSON_DIR = os.path.dirname(__file__)
print(f"[INFO] Loading JSON files from: {JSON_DIR}")

def is_calculation_question(question: str) -> bool:
    keywords = ["total", "sum", "how many", "count", "average", "mean", "min", "max", "minimum", "maximum"]
    return any(k in question.lower() for k in keywords)

def json_answer(question: str, data: dict) -> str:
    question = question.lower()

    if "how many volunteers" in question or "count volunteers" in question:
        return f"There are {len(data['volunteers'])} volunteers."

    elif "how many kits" in question or "count kits" in question:
        return f"There are {len(data['kits'])} kits distributed."

    elif "how many shifts" in question or "count shifts" in question:
        return f"There are {len(data['shifts'])} shifts recorded."

    elif "total hours" in question or "sum hours" in question:
        total_hours = sum(float(s.get("hours", 0)) for s in data["shifts"])
        return f"Total hours worked across all shifts: {total_hours}"

    elif "average hours" in question or "mean hours" in question:
        hours = [float(s.get("hours", 0)) for s in data["shifts"] if "hours" in s]
        avg_hours = sum(hours) / len(hours) if hours else 0
        return f"Average hours per shift: {avg_hours:.2f}"

    elif "min hours" in question or "minimum hours" in question:
        hours = [float(s.get("hours", 0)) for s in data["shifts"] if "hours" in s]
        return f"Minimum hours in a shift: {min(hours):.2f}" if hours else "No data available."

    elif "max hours" in question or "maximum hours" in question:
        hours = [float(s.get("hours", 0)) for s in data["shifts"] if "hours" in s]
        return f"Maximum hours in a shift: {max(hours):.2f}" if hours else "No data available."

    return "Sorry, I couldn't compute that from the available data."

def hybrid_qa(question: str, rag_chain, data: dict) -> str:
    return json_answer(question, data) if is_calculation_question(question) else rag_chain.run(question)

def build_rag_chain():
    profiles = {}

    try:
        with open(os.path.join(JSON_DIR, 'volunteers.json'), 'r', encoding='utf-8') as f:
            volunteers = json.load(f)
        print(f"Loaded {len(volunteers)} volunteers")

        with open(os.path.join(JSON_DIR, 'kits.json'), 'r', encoding='utf-8') as f:
            kits = json.load(f)
        print(f"Loaded {len(kits)} kits")

        with open(os.path.join(JSON_DIR, 'shifts.json'), 'r', encoding='utf-8') as f:
            shifts = json.load(f)
        print(f"Loaded {len(shifts)} shifts")

        with open(os.path.join(JSON_DIR, 'personal_stories.json'), 'r', encoding='utf-8') as f:
            stories = json.load(f)
        print(f"Loaded {len(stories)} stories")

        signups_path = os.path.join(JSON_DIR, 'signups_data.json')
        signups = []
        if os.path.exists(signups_path):
            with open(signups_path, 'r', encoding='utf-8') as f:
                signups = json.load(f)
            print(f"Loaded {len(signups)} signups")
        else:
            print("signups_data.json not found. Skipping signups.")

        for v in volunteers:
            vid = v.get('volunteer_id')
            if not vid:
                print(f"[WARN] Volunteer entry missing volunteer_id: {v}")
                continue
            profiles[vid] = {
                "profile": f"Volunteer Profile:\nID: {vid}\nName: {v.get('first_name')} {v.get('last_name')}\nDOB: {v.get('dob')}\nEmail: {v.get('email')}\nAddress: {v.get('address')}\nTitle: {v.get('title')}\nDays Available: {v.get('days_available')}",
                "kits": [], "shifts": [], "stories": [], "signups": []
            }

        for k in kits:
            vid = k.get('volunteer_id')
            if vid in profiles:
                profiles[vid]["kits"].append(
                    f"Kit ID: {k.get('kit_id')}, Type: {k.get('kit_type')}, Quantity: {k.get('quantity')}, Date: {k.get('date')}, Location: {k.get('location')}"
                )

        for s in shifts:
            try:
                vid = s['volunteer_id']
                if vid in profiles:
                    profiles[vid]["shifts"].append(
                        f"Shift ID: {s.get('shift_id')}, Title: {s.get('title')}, Hours: {s.get('hours')}, Date: {s.get('date')}"
                    )
            except KeyError as e:
                print(f"[SHIFT ERROR] Missing field {e} in: {s}")

        for story in stories:
            vid = story.get('volunteer_id')
            if vid in profiles:
                profiles[vid]["stories"].append(
                    f"Story ID: {story.get('story_id')}, Related Shift: {story.get('related_shift_id')}, Related Kit: {story.get('related_kit_id')}, Text: {story.get('text')}"
                )

        for signup in signups:
            pseudo_id = f"signup_{signup.get('id')}"
            profiles[pseudo_id] = {
                "profile": f"Signup:\nID: {signup.get('id')}\nName: {signup.get('name')}\nEmail: {signup.get('email')}\nPhone: {signup.get('phone')}\nShift: {signup.get('shift_title')}, Date: {signup.get('shift_date')}, Hours: {signup.get('shift_hours')}\nSigned at: {signup.get('created_at')}",
                "kits": [], "shifts": [], "stories": [], "signups": []
            }

        print(f"Constructed {len(profiles)} volunteer and signup profiles.")

    except Exception as e:
        print(f"[ERROR] Failed to load or parse JSON data: {e}")
        return None, {}

    documents = []
    for vid, data in profiles.items():
        try:
            total_hours = sum(
                float(s.split("Hours:")[1].split(",")[0].strip())
                for s in data["shifts"]
                if "Hours:" in s
            )
        except Exception as e:
            print(f"[WARN] Error calculating hours for {vid}: {e}")
            total_hours = 0

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
        template="""You are a helpful assistant managing volunteer data.

Use ONLY the context below.
If the question involves shifts or kits, consider totals, averages and distributions.
If info not present, reply: "I don't know."

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

    print("RAG chain initialized successfully.")
    return qa_chain, {
        "volunteers": volunteers,
        "kits": kits,
        "shifts": shifts,
        "signups": signups
    }

if __name__ == "__main__":
    rag_chain, data = build_rag_chain()
    if rag_chain:
        while True:
            question = input("\nAsk a question (type 'exit' to quit): ")
            if question.strip().lower() in ["exit", "quit"]:
                break
            answer = hybrid_qa(question, rag_chain, data)
            print("Answer:", answer)
