import os

from transformers import pipeline
import scipy

class MusicModel:
    def __init__(self):
        try:
            self.__synth = pipeline("text-to-audio", "facebook/musicgen-small")
            self.__file = "EmoMaestro.wav"

        except Exception as e:
            print(f"MusicModel Error: {e}")
            self.__model = None

    def gen(self, prompt: str, dur_sec: int = 30) -> str:
        try:
            music = self.__synth(text_inputs = prompt, forward_params = {"do_sample": True})

            path = self.__file

            scipy.io.wavfile.write(self.__file, rate = music["sampling_rate"], data = music["audio"])

            return path

        except Exception as e:
            print(f"MusicModel Error: {e}")

        return str()