from flask import Flask, jsonify

# Create the application instance
app = Flask(__name__)

@app.route("/api/home", methods=["GET"])
def home():
    return jsonify({
        "message": "Hello world!"
    })

if __name__ == "__main__":
    app.run(debug=True)
