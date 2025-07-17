# flask_app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_chain import build_rag_chain, hybrid_qa
import os

app = Flask(__name__)
CORS(app)

# Initialize RAG chain and JSON data
qa_chain, data_dict = build_rag_chain()

if not qa_chain or not data_dict:
    raise RuntimeError("❌ Failed to initialize RAG chain or load JSON data.")

print("✅ RAG chain initialized and ready.")

@app.route("/api/shifts", methods=["GET"])
def get_shifts():
    return jsonify(data_dict.get("shifts", []))

@app.route("/api/kits", methods=["GET"])
def get_kits():
    return jsonify(data_dict.get("kits", []))

@app.route("/api/volunteers", methods=["GET"])
def get_volunteers():
    return jsonify(data_dict.get("volunteers", []))

@app.route("/api/signups", methods=["GET"])
def get_signups():
    return jsonify(data_dict.get("signups", []))

@app.route("/api/personal-stories", methods=["GET"])
def get_stories():
    return jsonify(data_dict.get("stories", []))

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Missing question."}), 400

        print(f"[QUERY] {question}")
        answer = hybrid_qa(question, qa_chain, data_dict)
        print(f"[ANSWER] {answer}")

        return jsonify({"answer": answer})
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": str(os.times())})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use dynamic port on Render
    app.run(host="0.0.0.0", port=port, threaded=True)
