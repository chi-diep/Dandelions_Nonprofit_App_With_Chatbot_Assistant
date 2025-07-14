import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_chain import build_rag_chain, hybrid_qa

app = Flask(__name__)
CORS(app)

qa_chain, conn = build_rag_chain()

def load_json_data(filename):
    try:
        with open(os.path.join("Data- JSON format", filename), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        return []

@app.route("/api/shifts")
def get_shifts():
    data = load_json_data("shifts.json")
    return jsonify(data)

@app.route("/api/kits")
def get_kits():
    data = load_json_data("kits.json")
    return jsonify(data)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        print(f"Incoming question: {question}")

        if not question:
            return jsonify({"error": "Missing question."}), 400

        answer = "I don't know."
        if qa_chain and conn:
            answer = hybrid_qa(question, qa_chain, conn)
        else:
            print("RAG chain or DB connection not initialized.")

        print(f"Answer: {answer}")
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
