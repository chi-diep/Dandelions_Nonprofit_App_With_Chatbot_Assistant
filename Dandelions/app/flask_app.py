# flask_app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_chain import build_rag_chain, hybrid_qa
import json
import os

app = Flask(__name__)
CORS(app)

# load JSON data into memory
data_dict = {
    "volunteers": [],
    "kits": [],
    "shifts": [],
    "signups": [],
    "stories": []
}

try:
    with open('Data- JSON format/volunteers.json', 'r') as f:
        data_dict["volunteers"] = json.load(f)
    with open('Data- JSON format/kits.json', 'r') as f:
        data_dict["kits"] = json.load(f)
    with open('Data- JSON format/shifts.json', 'r') as f:
        data_dict["shifts"] = json.load(f)
    with open('Data- JSON format/signups_data.json', 'r') as f:
        data_dict["signups"] = json.load(f)
    with open('Data- JSON format/personal_stories.json', 'r') as f:
        data_dict["stories"] = json.load(f)
    print("Loaded all JSON data into memory.")
except Exception as e:
    print(f"Failed to load JSON data: {e}")

# initialize RAG chain
qa_chain, _ = build_rag_chain()

@app.route("/api/shifts")
def get_shifts():
    return jsonify(data_dict["shifts"])

@app.route("/api/kits")
def get_kits():
    return jsonify(data_dict["kits"])

@app.route("/api/volunteers")
def get_volunteers():
    return jsonify(data_dict["volunteers"])

@app.route("/api/signups")
def get_signups():
    return jsonify(data_dict["signups"])

@app.route("/api/personal-stories")
def get_stories():
    return jsonify(data_dict["stories"])

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        print(f"Incoming question: {question}")

        if not question:
            return jsonify({"error": "Missing question."}), 400

        answer = hybrid_qa(question, qa_chain, data_dict)
        print(f"Answer: {answer}")
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": str(os.times())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
