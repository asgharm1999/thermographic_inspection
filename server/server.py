from flask import Flask, jsonify
from flask_cors import CORS

# Create the application instance
app = Flask(__name__)
CORS(app)


@app.route("/api/home", methods=["GET"])
def home():
    return jsonify({"message": "If you see this, the backend is working!"})


if __name__ == "__main__":
    # Use port 8080 because default port 5000 doesn't work with CORS
    app.run(debug=True, port=8080)
