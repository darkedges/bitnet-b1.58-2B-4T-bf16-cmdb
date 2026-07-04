from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Allows the HTML file to talk to this script

FEEDBACK_FILE = "cmdb_feedback_loop.csv"


@app.route("/log_feedback", methods=["POST"])
def log_feedback():
    data = request.json

    # Flatten the data for CSV
    row = {
        "Timestamp": data["timestamp"],
        "App": data["app_name"],
        "Avail": data["attributes"]["avail"],
        "Sens": data["attributes"]["sens"],
        "Classif": data["attributes"]["classif"],
        "AI_Rec": data["ai_recommendation"],
        "Is_Correct": data["is_correct"],
        "Correction": data["correction"],
    }

    df = pd.DataFrame([row])
    df.to_csv(
        FEEDBACK_FILE, mode="a", index=False, header=not os.path.exists(FEEDBACK_FILE)
    )

    return jsonify({"status": "success"})


if __name__ == "__main__":
    print(f"Feedback server running. Logs saving to {FEEDBACK_FILE}")
    app.run(port=5000)
