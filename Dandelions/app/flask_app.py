# flask_app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from rag_chain import build_rag_chain, hybrid_qa

app = Flask(__name__)
CORS(app)

# set up the RAG and hybrid chain right away
qa_chain, conn = build_rag_chain()

import os

def get_db_connection():
    return psycopg2.connect(
        dbname="Dandelions",
        user="postgres",
        password=os.getenv("DB_PASSWORD"),
        host="localhost",
        port="5432"
    )

@app.route("/api/shifts")
def get_shifts():
    try:
        # get all the shift data from the database
        con = get_db_connection()
        cur = con.cursor()
        cur.execute("SELECT volunteer_id, shift_id, title, hours, date FROM shifts;")
        rows = cur.fetchall()
        cur.close()
        con.close()
        data = [
            {
                "volunteer_id": r[0],
                "shift_id": r[1],
                "title": r[2],
                "hours": float(r[3]),
                "date": str(r[4])
            }
            for r in rows
        ]
        return jsonify(data)
    except Exception as e:
        print(f"Error fetching shifts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/kits")
def get_kits():
    try:
        # get all the kits data from the database
        con = get_db_connection()
        cur = con.cursor()
        cur.execute("SELECT volunteer_id, kit_id, kit_type, quantity, date_given, location FROM kits;")
        rows = cur.fetchall()
        cur.close()
        con.close()
        data = [
            {
                "volunteer_id": r[0],
                "kit_id": r[1],
                "kit_type": r[2],
                "quantity": int(r[3]),
                "date": str(r[4]),
                "location": r[5]
            }
            for r in rows
        ]
        return jsonify(data)
    except Exception as e:
        print(f"Error fetching kits: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        # handle question sent from the frontend
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
