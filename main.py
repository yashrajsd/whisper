from flask import Flask,request,jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")


@app.route("/")
def Home():
    return "Working"

@app.route("/",methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    file_path = "/tmp/audio.wav"
    file.save(file_path)

    segments, _ = model.transcribe(file_path)
    text = " ".join(segment.text for segment in segments)
    return jsonify({"transcription": text})

if __name__ == '__main__':
    app.run()