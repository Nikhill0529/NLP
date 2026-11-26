Practical No. 1: NLTK, Text-to-Speech & Speech-to-Text
#1(a) Install NLTK (Command Prompt madhe run karayche)

pip install --upgrade pip
pip install --user -U nltk
pip install --user -U numpy

#IDLE / Python shell madhe verify karnyasathi:

import nltk
nltk.download('punkt')









#1(b) Text to Speech (p1b.py) — Prerequisite: pip install gTTS playsound==1.2.2

from gtts import gTTS
from playsound import playsound

mytext = "Welcome to Natural Language programming"
language = "en"
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("myfile.mp3")
playsound("myfile.mp3")









#1(c) Speech to Text

import speech_recognition as sr

filename = "male.wav"
r = sr.Recognizer()
with sr.AudioFile(filename) as source:
    audio_data = r.record(source)
    text = r.recognize_google(audio_data)
    print(text)

