import os
import json
import librosa
import numpy as np
import pandas as pd
import librosa.display
from scipy.io.wavfile import write
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_ = 67
class WordTokenizer:
    def __init__(self, words_path: str, texts_path: str, max_len: int):
        self.max_len = max_len
        self.texts_path = texts_path
        words_file = open(words_path, "rb")
        self.words = list(set(words_file.read().decode('utf-8').replace("،", "").replace("؟", "").split(" ")))
        self.word_index = None
        self.words_mat = None
        words_file.close()

    def tokenizer(self):
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts([self.words])
        self.word_index = tokenizer.word_index

        return self.word_index

    def convert_mat(self):
        self.tokenizer()
        text_path_list = os.listdir(self.texts_path)
        zer = np.zeros((len(text_path_list), self.max_len))

        for i, f_t in enumerate(text_path_list):
            with open(f"{self.texts_path}/{f_t}", "rb") as text_file:
                plain_text = text_file.read().decode("utf-8").replace("؟", "").replace("،", "").strip().split(" ")
                arr_text = [self.word_index[i] for i in plain_text]
                zer[i][:len(arr_text)] = arr_text


        self.words_mat = zer
        return self.words_mat

    def save(self):
        self.convert_mat()
        word_index = open("word_index.txt", "wb")
        word_index.write(str(self.word_index).encode("utf-8"))
        word_index.close()

        df = pd.DataFrame(self.words_mat, columns=[f"WR#{i}" for i in range(self.max_len)])
        df.to_csv("WordsMatrix.csv", index=False)


    def convertor(self, sam):
        if not os.path.exists("WordsMatrix.csv"):
            raise FileNotFoundError("Please call save method first!")

        word_index_file = open("word_index.txt", "rb")
        word_index = json.loads(word_index_file.read().decode("utf-8").replace("'", '"'))
        sam = sam[sam != 0]

        sentence = ""
        for i, s in enumerate(sam):
            if i == 0:
                sentence += list(word_index.keys())[list(word_index.values()).index(int(s))]
            else:
                sentence += f" {list(word_index.keys())[list(word_index.values()).index(int(s))]}"

        print(sentence)


sample = WordTokenizer("output.txt", "MainDataset/maletexts", max_)

class PrepareAudios:
    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self.audio_list = os.listdir(audio_path)
        self.target_sample_rate = 16000
        self.target_length = 42000
        self.n_mels = 128
        self.n_fft = 4080
        self.hop_length = 256

    def convert_to_mat(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.target_sample_rate)

        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
        else:
            audio = audio[:self.target_length]

        spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        return spectrogram_db

    def convert_to_audio(self, spectrogram, file_path):
        power_spectrogram = librosa.db_to_power(spectrogram)
        reconstructed_stft = librosa.feature.inverse.mel_to_stft(
            power_spectrogram,
            sr=self.target_sample_rate,
            n_fft=self.n_fft
        )
        audio = librosa.griffinlim(
            reconstructed_stft,
            hop_length=self.hop_length,
            n_iter=24
        )

        audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
        write(file_path, self.target_sample_rate, audio)

        return audio

    def fit(self):
        audios = []

        for file in self.audio_list:
            test_file = os.path.join(self.audio_path, file)
            mat = self.convert_to_mat(test_file)
            audios.append(mat)

        audios = np.array(audios)
        return audios