import whisper
from pydub import AudioSegment
from pydub.utils import make_chunks
import spacy
import warnings
import re
import os
import shutil

class Whisper_transcripts:
    def __init__(self, spacy_model="en_core_web_sm"):
        warnings.filterwarnings("ignore",category=FutureWarning)
        self.nlp=spacy.load(spacy_model)
        self.whisper_model=whisper.load_model("base")
    
    def transcribe(self,filepath):
        audio=AudioSegment.from_file(filepath)
        chunks=make_chunks(audio, 30000)
        chunks_dir="chunks"
        os.makedirs(chunks_dir, exist_ok=True)
        transcript=""
        for i, chunk in enumerate(chunks):
            chunk_filename=os.path.join(chunks_dir, f"chunk{i}.wav")
            chunk.export(chunk_filename, format="wav")
            result=self.whisper_model.transcribe(chunk_filename)
            transcript+=result['text']+" "
        shutil.rmtree(chunks_dir)
        return transcript
    
    def process_audio_video(self, filepath):
        text=self.transcribe(filepath)
        text=re.sub(r'\s+',' ',text)
        text=re.sub(r'[^\w\s,!?]','',text)
        text=re.sub(r'([.,?])\1+',r'\1',text)

        return text