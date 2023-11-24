from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from io import BytesIO

sys.path.insert(
    0, "C:/Users/chris/OneDrive/Documents/GitHub/thermographic_inspection/preprocess"
)
from preprocessFuncs import preprocess


app = Flask(__name__)
CORS(app)

# Global variables
coldPath = None
hotPath = None


@app.route("/upload", methods=["POST", "GET"])
def uploadFile():
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded"})

    file = request.files["file"]
    fileName = file.filename
    videoType = request.form["type"]

    if fileName == "":
        return jsonify({"message": "No file selected"})

    # Check if file is video
    if file.filename.split(".")[-1] not in ["mp4", "avi"]:
        return jsonify({"message": "File is not a video"})

    file.save("temp/" + file.filename)
    if videoType == "cold":
        global coldPath
        coldPath = "temp/" + file.filename
    else:
        global hotPath
        hotPath = "temp/" + file.filename

    return jsonify({"message": "File uploaded"})


@app.route("/preprocess", methods=["POST", "GET"])
def preprocessVideo():
    if coldPath is None:
        return jsonify({"message": "Cold video not uploaded"})
    elif hotPath is None:
        return jsonify({"message": "Hot video not uploaded"})
    
    resultPath = preprocess(coldPath, hotPath, "client/public/", method="PCT")
    resultPath = resultPath.removeprefix("client/public/")

    return jsonify({"message": "Video preprocessed", "resultPath": resultPath})


if __name__ == "__main__":
    app.run(debug=True, port=8080)
