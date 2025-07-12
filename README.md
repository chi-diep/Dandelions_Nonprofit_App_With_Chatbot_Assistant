
# Volunteer Management & AI Assistant  
*A web app with hybrid RAG pipeline using Flask, PostgreSQL, LangChain, Ollama, and HuggingFace embeddings*

## Why This Project Matters

Organizations like nonprofits and community centers handle hundreds of volunteers, shifts, and kit distributions. Tracking who did what, how many hours they served, and sharing stories is often manual and time consuming.

**What if we had an AI assistant that could instantly answer questions like:**
- “How many volunteers have signed up?”
- “What’s the total number of kits given out?”
- “Tell me a story from a volunteer who worked more than 10 hours.”

By building a Retrieval-Augmented Generation (RAG) system tied directly to a volunteer database, this app makes it possible to pull precise insights or rich narratives instantly.

This helps small organizations manage volunteers better, highlight impact stories, and run operations more efficiently.

## What Was the Data?

**Database:** PostgreSQL with five main tables:

- `volunteers`  
- `shifts`  
- `kits`  
- `personal_stories`  -> This is updated once volunteers sign up for shifts
- `signups`

Sample structure:
Let's build both traditional SQL aggregations (like `COUNT`, `SUM`, `MAX`) and a semantic RAG system to pull richer context.

## How Does the Hybrid RAG Work?

1. **Data Loading**  
   - Extracts volunteers, shifts, kits, stories from PostgreSQL.
   - Merges them into natural language chunks (profile + total hours + stories).

2. **Embeddings & Vector Store**  
   - Uses Hugging Face MiniLM to embed the chunks.
   - Stores vectors in ChromaDB.

3. **Language Models**  
   - Runs local language models (like Mistral) using LangChain Ollama.
   - For questions needing counting / sums (e.g. “total volunteers”), does direct SQL.

4. **Smart Classifier**  
   - If question involves totals, counts, or aggregates, it triggers SQL.  
   - Otherwise, it uses RAG to retrieve context and generate answers.

## Example Queries It Can Handle

| Type                  | Example                                       |
|------------------------|-----------------------------------------------|
| Counts / Totals        | “How many volunteers are in the database?”    |
| Summaries              | “Who worked the most hours?”                  |
| Stories & Context      | “Share a story from someone who gave kits.”   |
| Kit Distributions      | “How many kits were given out?”               |

## What Could This Look Like in Real Life?

- Nonprofits seeing real-time volunteer stats without Excel.
- Local managers pulling hours to prepare impact reports.
- Donor coordinators fetching stories tied to actual volunteer IDs.

## Technologies Used

- Python  
- Flask  
- PostgreSQL  
- ChromaDB  
- LangChain & LangChain Ollama  
- Hugging Face sentence-transformers  
- Local models: Mistral, TinyLLaMA  
- JavaScript + Tailwind CSS for frontend  
- AOS animations & draggable chat widget

## What Challenges Came Up?

- Local models initially gave irrelevant or made-up answers (“hallucinations”).
- Had to experiment with chunk sizes, number of retrieved docs, and prompt templates.
- Optimized by:
  - Using smaller models to fit local resources.
  - Explicitly instructing LLM to admit “I don’t know” if uncertain.
  - Fine-tuning thresholds for hybrid switching between SQL vs RAG.

## How to Run This Project

Follow these steps to set up and run the volunteer management & AI assistant app on your machine.

1. Clone the Repo
```bash
git clone https://github.com/yourusername/dandelions-volunteer-ai.git
cd dandelions-volunteer-ai
```
2. Install Python Dependencies
Make sure you have Python 3.10+ installed.
Then install the required Python packages.
```bash
pip install -r requirements.txt
```
3. Set Up PostgreSQL Database
Ensure PostgreSQL is installed and running.
Create a database called Dandelions.
```
CREATE DATABASE Dandelions;
```
Create the tables (volunteers, kits, shifts, personal_stories, signups) and populate them.
You can use your own data or a provided SQL file.

Example to load:

```bash
psql -U postgres -d Dandelions -f setup.sql
(Adjust -U user and password as needed.)
```

4. Start the Flask Backend
Run your hybrid RAG + SQL API server:
```bash
python flask_app.py
This starts on http://localhost:5000 and serves:

/api/shifts → JSON list of shifts

/api/kits → JSON list of kits

/ask → POST endpoint for natural language questions
```

5. Run the Frontend
```
Option A: Using Live Server (VSCode extension or similar)
Open the project folder in VSCode, right-click on index.html and select “Open with Live Server.”

Option B: Python HTTP server
bash
python -m http.server
Then open http://localhost:8000 in your browser.
```
6. Try It Out!
See metrics auto-count from /api/shifts and /api/kits. Click the “Need help?” chat bubble and ask questions like:
```
How many volunteers are there?
Show me stories from volunteers.
How many kits were delivered?
Troubleshooting

If the AI says “I don’t know” to count questions:
Make sure your database is running and your flask_app.py is connected correctly.

If the metrics don’t show up:
Confirm /api/shifts and /api/kits return data by visiting http://localhost:5000/api/shifts in your browser.

If Flask says port is in use:
Stop other apps on port 5000 or change it inside flask_app.py.
```


