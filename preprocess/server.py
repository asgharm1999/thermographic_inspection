from flask import Flask, request, jsonify
from flask_cors import CORS
from preprocessFuncs import preprocess
import json

# Create Flask server
app = Flask(__name__)
CORS(app)

# Global variables
coldPath = None
hotPath = None

@app.route("/upload", methods=["POST", "GET"])
def uploadFile():
    # Check if a file is in the request
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded"})
    
    # Get file, file name, and video type
    file = request.files["file"]
    fileName = file.filename
    videoType = request.form["type"]

    # Check if the file is empty
    if fileName == "":
        return jsonify({"message": "No file selected"})
    
    # Check if the file is a video
    if file.filename.split(".")[-1] not in ["mp4", "avi"]:
        return jsonify({"message": "File is not a video"})
    
    # Save the file
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
    # Check if both videos are uploaded
    if coldPath is None:
        return jsonify({"message": "Cold video not uploaded"})
    elif hotPath is None:
        return jsonify({"message": "Hot video not uploaded"})
    
    # Get the method
    method = request.form["method"]
    options = request.form["options"]
    options = json.loads(options)

    print(options)
    
    # Preprocess the videos
    resultPath = preprocess(coldPath, hotPath, "client/public/", method=method, options=options)
    resultPath = resultPath.removeprefix("client/public/")

    return jsonify({"message": "Video preprocessed", "resultPath": resultPath})


if __name__ == "__main__":
    app.run(debug=True, port=8080)  # CORS does not work with default port 5000

