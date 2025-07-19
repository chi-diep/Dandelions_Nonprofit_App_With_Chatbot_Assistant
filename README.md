# 🌼 Dandelions: Volunteer Management & AI Assistant  
*A student-built web app with RAGFlow, JSON data, Streamlit, and Ollama for local AI querying*

## Why This Project Matters

Many nonprofits and community organizations handle volunteers using scattered spreadsheets and forms. Tracking hours, signups, kit distributions, and stories takes time and effort.

**What if you could just ask:**
- “How many kits were distributed in total?”
- “Tell me a story from a volunteer who worked more than 10 hours.”
- “Who signed up for shifts on Saturday?”

This app uses **Retrieval-Augmented Generation (RAG)** to turn static JSON data into dynamic insights, powered by a local language model — no internet or database required.

## Project Overview

- All data is loaded from local `.json` files (volunteers, kits, shifts, stories, signups).
- A profile is created for each volunteer or signup.
- The app embeds this data using HuggingFace and stores it in ChromaDB.
- It uses **Ollama** to run a local LLM like Mistral.
- You interact with the system through a **Streamlit chatbot UI**.

## What the AI Assistant Can Handle

| Type                  | Example                                         |
|------------------------|-------------------------------------------------|
| Count / Sum            | “How many volunteers signed up?”               |
| Kit Insights           | “How many kits did Shannon give out?”          |
| Shift Analysis         | “Who worked more than 4 hours?”                |
| Stories                | “Tell me a story from a volunteer in May.”     |
| Profiles               | “What is the ID of Shannon Hamilton?”          |

## Tech Stack

- **Streamlit** – for an interactive chatbot UI  
- **RAGFlow** – custom hybrid RAG pipeline  
- **Ollama** – for running Mistral or other local LLMs  
- **ChromaDB** – for storing vector embeddings  
- **HuggingFace Embeddings** – MiniLM for fast, local vectorization  
- **JSON** – structured data, no SQL setup required  
- **Python** – core application logic

