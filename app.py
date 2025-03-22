import os
from OCR_processing import OCR_processing
from image_text_extraction import Image_text_Extraactor
from langchain_trial import Chatbot_response
from qdrant_chunking import QdrantChunking
from qdrant_retrieval import Qdrant_retrieval
from whisper_transcripts import Whisper_transcripts
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify,send_from_directory
import speech_recognition as sr

class FileProcessor:
    def __init__(self, folder):
        self.upload_folder=folder
        os.makedirs(folder, exist_ok=True)
        self.ocr=OCR_processing()
        self.image_processing=Image_text_Extraactor()
        self.transcripts=Whisper_transcripts()
    
    def save_file(self, files):
        uploaded_file_paths=[]
        for file in files:
            filename=file.filename
            filepath=os.path.join(self.upload_folder,filename)
            if not os.path.exists(filepath):
                file.save(filepath)
                uploaded_file_paths.append(filepath)
        return uploaded_file_paths
            
    def process_files(self,filepaths):
        text=""
        for file in filepaths:
            extension=os.path.splitext(file)[1].lower()
            if extension in (".mp4",".mp3",".wav"):
                text+=self.transcripts.process_audio_video(file)
            elif extension in (".doc",".docx"):
                text+=self.ocr.doc_text(file)
            elif extension==".pdf":
                text+=self.ocr.pdf_text
            elif extension in (".png",".jpg",".jpeg"):
                text+=self.image_processing.image_extraction(file)
        self.save_text_file(file, text)

    def save_text_file(self, file, text):
        base_name=os.path.basename(file)
        filename_no_ext=os.path.splitext(base_name)[0]
        text_file=f"{filename_no_ext}.txt"
        path=os.path.join(self.upload_folder, text_file)
        with open(path, "w") as file:
            file.write(text)

app=Flask(__name__)
UPLOAD_FOLDER = ["uploads"]
file_processor = FileProcessor(UPLOAD_FOLDER[0])
qdrant_chunking = QdrantChunking()
collection_name="test_check"
query_processor=Qdrant_retrieval(collection_name)
graph_runner=Chatbot_response()
r = sr.Recognizer()

@app.route("/",methods=["GET","POST"])
def home():
    if request.method=="POST":
        files=request.files.getlist("files")
        uploaded_file_paths=file_processor.save_file(files)
        if uploaded_file_paths:
            file_processor.process_files(uploaded_file_paths)
            qdrant_chunking.builder_graph(UPLOAD_FOLDER[0],collection_name)
        return redirect(url_for("home"))
    folders=UPLOAD_FOLDER
    transcript_files=[]
    for i in folders:
        transcript_files.extend([f for f in os.listdir(i) if f.endswith(".txt")])
    return render_template("index.html",transcript_files=transcript_files)

@app.route("/download/<filename>")
def download_file(filename):
    for folder in UPLOAD_FOLDER:
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
    
    flash("File not found!", "error")
    return redirect(url_for("home"))

@app.route("/view/<filename>")
def view_file(filename):
    # Search for the file in UPLOAD_FOLDERS
    for folder in UPLOAD_FOLDER:
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            return send_from_directory(folder, filename)

    flash("File not found!", "error")
    return redirect(url_for("home"))

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/chatbot/query", methods=["POST"])
def chatbot_query():
    try:
        role=request.json.get("role")
        user_input = request.json.get("user_input")
        print(f"role is {role}")
        if not user_input:
            return jsonify({"error": "Invalid input"}), 400

        # Process user query through DissertationQueryProcessor
        retrieved_chunks = query_processor.qdrant_retrieve(user_input)
        input_state = {
        "retreived_chunks": retrieved_chunks,
        "role": role,
        "model": "mistral"
    }
        response=graph_runner.run(input_state)
        return jsonify({"response": response["rephrased_chunks"]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def clear_files():
    directories=["uploads"]
    set=False
    for directory in directories:
        try:
            set=True
            if not os.path.exists(directory):
                print(f"Directory {directory} path is incorrect or does not exist")
                set=False
            if not os.path.isdir(directory):
                print(f"Directory {directory} is not a directory")
                set=False
            for file in os.listdir(directory):
                print(file)
                file_path=os.path.join(directory,file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    print(f"Skipping {file_path} for deletion")
            set=True
        except Exception as e:
            print(f"An error occurred: {e}")
            set= False
    if set==True:
        return True
    else:
        return False

@app.route("/clear")
def delete_all_files():
    success=clear_files()
    if success:
        return redirect(url_for("home"))
    else:
        return "Failed to delete Files",500
    
@app.route('/chatbot/voice', methods=['POST'])
def voice_input():
    try:
        # Use the microphone as input
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.2)
            print("Listening for voice input...")
            audio2 = r.listen(source2)

            # Convert speech to text
            MyText = r.recognize_google(audio2).lower()
            print(f"Recognized: {MyText}")

            if MyText == "exit":
                return jsonify({'transcription': '', 'message': 'Exit command received'})

            return jsonify({'transcription': MyText})

    except sr.RequestError:
        return jsonify({'error': 'Speech Recognition API unavailable'}), 500
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio'}), 400

if __name__ == "__main__":
    app.run(debug=True)

