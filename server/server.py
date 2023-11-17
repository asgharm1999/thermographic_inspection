from flask import Flask, request, jsonify
from flask_cors import CORS
import sys

sys.path.insert(
    0, "C:/Users/chris/OneDrive/Documents/GitHub/thermographic_inspection/preprocess"
)
from preprocessFuncs import preprocess


app = Flask(__name__)
CORS(app)


@app.route("/upload", methods=["POST", "GET"])
def processFile():
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"message": "No file selected"})

    # Check if file is video
    if file.filename.split(".")[-1] not in ["mp4", "avi"]:
        return jsonify({"message": "File is not a video"})

    return jsonify({"message": "File uploaded"})

    # # Save file
    # file.save("videos/" + file.filename)

    # # Preprocess file
    # ppf.preprocess("videos/" + file.filename, method="EOF", numEOFs=6)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
