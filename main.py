"""
Blueprint Trader - Trading Strategy Engine
"""

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "name": "Blueprint Trader",
        "description": "Trading strategy and analysis engine"
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    print("Starting Blueprint Trader...")
    app.run(host="0.0.0.0", port=5000, debug=False)
