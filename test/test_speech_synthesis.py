from gtts import gTTS
import os


def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    filename = "speech.mp3"
    tts.save(filename)
    os.system(f"start {filename}")


if __name__ == "__main__":
    text = "Hello, I am a speech synthesis system created with Python."
    text_to_speech(text)
