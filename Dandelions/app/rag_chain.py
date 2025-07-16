"""
This script shows an alternate way to implement RAG (Retrieval-Augmented Generation)
by reading data from JSON files instead of a database.

It supports:
- JSON-based direct calculations for questions involving count, total, average, max, min.
- RAG-based contextual answers using Ollama + Chroma vector store for other questions.
"""

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

        for v in volunteers:
            vid = v['volunteer_id']
            profiles[vid] = {
                "profile": f"Volunteer Profile:\nID: {vid}\nName: {v['first_name']} {v['last_name']}\nDOB: {v['dob']}\nEmail: {v['email']}\nAddress: {v['address']}\nTitle: {v['title']}\nDays Available: {v['days_available']}",
                "kits": [], "shifts": [], "stories": [], "signups": []
            }

        for k in kits:
            vid = k['volunteer_id']
            if vid in profiles:
                profiles[vid]["kits"].append(
                    f"Kit ID: {k['kit_id']}, Type: {k['kit_type']}, Quantity: {k['quantity']}, Date: {k['date']}, Location: {k['location']}"
                )

        for s in shifts:
            vid = s['volunteer_id']
            if vid in profiles:
                profiles[vid]["shifts"].append(
                    f"Shift ID: {s['shift_id']}, Title: {s['title']}, Hours: {s['hours']}, Date: {s['date']}"
                )

        for story in stories:
            vid = story['volunteer_id']
            if vid in profiles:
                profiles[vid]["stories"].append(
                    f"Story ID: {story['story_id']}, Related Shift: {story['related_shift_id']}, Related Kit: {story['related_kit_id']}, Text: {story['text']}"
                )

        for signup in signups:
            pseudo_id = f"signup_{signup['id']}"
            profiles[pseudo_id] = {
                "profile": f"Signup:\nID: {signup['id']}\nName: {signup['name']}\nEmail: {signup['email']}\nPhone: {signup['phone']}\nShift: {signup['shift_title']}, Date: {signup['shift_date']}, Hours: {signup['shift_hours']}\nSigned at: {signup['created_at']}",
                "kits": [], "shifts": [], "stories": [], "signups": []
            }

        print(f"✅ Loaded data: {len(profiles)} volunteer & signup profiles.")

    except Exception as e:
        print(f"❌ Failed to load JSON data: {e}")
        return None, {}

    # Build documents for RAG
    documents = []
    for vid, data in profiles.items():
        total_hours = sum(float(s.split("Hours:")[1].split(",")[0].strip())
                          for s in data["shifts"] if "Hours:" in s)
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

    print("✅ RAG chain fully initialized.")
    return qa_chain, {
        "volunteers": volunteers,
        "kits": kits,
        "shifts": shifts,
        "signups": signups
    }


def hybrid_qa(query, qa_chain, data_dict):
    query = query.lower()
    try:
        volunteers = data_dict.get("volunteers", [])
        kits = data_dict.get("kits", [])
        shifts = data_dict.get("shifts", [])
        signups = data_dict.get("signups", [])

        hours = [float(s.get("hours", 0)) for s in shifts if s.get("hours") is not None]
        signup_hours = [float(s.get("shift_hours", 0)) for s in signups if s.get("shift_hours") is not None]

        combined_hours = hours + signup_hours
        response = ""

        # COUNT and TOTAL
        if any(word in query for word in ["count", "total", "number of", "how many"]):
            if "volunteer" in query:
                response = f"[JSON] There are a total of {len(volunteers)} volunteers."
            elif "kit" in query:
                if "hygiene" in query:
                    count = sum(1 for k in kits if "hygiene" in k.get("kit_type", "").lower())
                    response = f"[JSON] There are {count} hygiene kits distributed."
                else:
                    response = f"[JSON] There are a total of {len(kits)} kits distributed."
            elif "shift" in query:
                if "food distribution" in query:
                    count = sum(1 for s in shifts if "food distribution" in s.get("title", "").lower())
                    response = f"[JSON] There are {count} 'Food Distribution' shifts recorded."
                else:
                    response = f"[JSON] There are a total of {len(shifts)} shifts recorded."
            elif "signup" in query:
                response = f"[JSON] There are a total of {len(signups)} volunteer signups."

        # SUM / TOTAL HOURS
        elif "total" in query or "sum" in query:
            if "hours" in query:
                total_hours = sum(combined_hours)
                response = f"[JSON] The total hours worked across all shifts and signups is {total_hours:.2f} hours."

        # AVERAGE HOURS
        elif "average" in query or "mean" in query:
            if combined_hours:
                avg = sum(combined_hours) / len(combined_hours)
                response = f"[JSON] The average hours per shift or signup is {avg:.2f} hours."
            else:
                response = "[JSON] No hours data available to compute average."

        # MAX HOURS
        elif "max" in query or "largest" in query or "most hours" in query:
            if combined_hours:
                response = f"[JSON] The maximum hours recorded in a shift or signup is {max(combined_hours)} hours."
            else:
                response = "[JSON] No hours data available to compute maximum."

        # MIN HOURS
        elif "min" in query or "smallest" in query or "least hours" in query:
            if combined_hours:
                response = f"[JSON] The minimum hours recorded in a shift or signup is {min(combined_hours)} hours."
            else:
                response = "[JSON] No hours data available to compute minimum."

        # TOP VOLUNTEER BY HOURS
        elif "top volunteer" in query or "most hours by volunteer" in query:
            from collections import defaultdict
            hour_map = defaultdict(float)
            for s in shifts:
                if s.get("volunteer_id") and s.get("hours"):
                    hour_map[s["volunteer_id"]] += float(s["hours"])
            if hour_map:
                top_id = max(hour_map, key=hour_map.get)
                top_hours = hour_map[top_id]
                name = next((f"{v['first_name']} {v['last_name']}" for v in volunteers if v["volunteer_id"] == top_id), top_id)
                response = f"[JSON] {name} worked the most hours: {top_hours:.2f} hours."
            else:
                response = "[JSON] No volunteer hour data available."

        # HOURS PER VOLUNTEER
        elif "hours per volunteer" in query:
            from collections import defaultdict
            hour_map = defaultdict(float)
            for s in shifts:
                if s.get("volunteer_id") and s.get("hours"):
                    hour_map[s["volunteer_id"]] += float(s["hours"])
            result_lines = []
            for vid, total in hour_map.items():
                name = next((f"{v['first_name']} {v['last_name']}" for v in volunteers if v["volunteer_id"] == vid), vid)
                result_lines.append(f"{name}: {total:.2f} hours")
            response = "[JSON] Hours per volunteer:\n" + "\n".join(result_lines) if result_lines else "[JSON] No data available."

        # return response if found
        if response:
            return response

    except Exception as e:
        print(f"Direct JSON analysis failed: {e}")

    print("Falling back to RAG...")
    rag_result = qa_chain.invoke({"query": query})
    return rag_result.get("result", "I don't know.")
