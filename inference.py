#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from typing import Text, Optional
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
from pyctcdecode import build_ctcdecoder
from pathlib import Path
import json
import os


class Inference:
    """
    parses the audio and outputs the `transcript` with or without a lnaguage model
    Parameters:
    ----------
    Methods:
    --------
    """
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_transcription(self, model_name: Text, audio_file: Text, lm_file: Text = None) -> Text:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        vocab_path = str(Path().absolute())
        with open(os.path.join(vocab_path, "./clu/speech/vocab.json")) as json_file:
            sorted_dict = json.load(json_file)
        decoder = build_ctcdecoder(list(sorted_dict.keys()), lm_file)
        if lm_file:
            audio_input, _ = librosa.load(audio_file, sr=16_000)
            input_values = processor(audio_input, sampling_rate=16_000, return_tensors="pt").input_values.to(self.device)
            with torch.no_grad():
                logits = model(input_values).logits.cpu().numpy()[0]
                output = decoder.decode(logits)
                return output
        elif lm_file == None:
            audio_input, _ = librosa.load(audio_file, sr=16_000)
            input_values = processor(audio_input, sampling_rate=16_000, return_tensors="pt").input_values.to(self.device)
            with torch.no_grad():
                logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1).squeeze()
                softmax_probs = torch.nn.Softmax(dim=-1)(logits)
                seq_prob = torch.max(softmax_probs, dim=-1).values.squeeze()
                rem_tokens = [0,1,2,3,4]
                seq_prob = seq_prob[[(token not in rem_tokens) for token in predicted_ids]]
                confidence = (torch.prod(seq_prob)**(1/len(seq_prob))).item()
                transcription = processor.decode(predicted_ids, skip_special_tokens=True)
                return transcription


class Transcription:
    def __init__(self, model_name: Text, allow_chunking: bool = False, lm_file: Optional[Text] = None) -> None:
        self.model_name = model_name
        self.allow_chunking = allow_chunking
        self.lm_file = lm_file
        self.inference = Inference()

    def transcribe(self, audio_file):
        """chunking, language model for both, punctuation restoration"""
        if self.allow_chunking == True:
            
            # see https://huggingface.co/blog/asr-chunking
            pipe = pipeline("automatic-speech-recognition", model=self.model_name, devide=self.inference.device)
            output = pipe(audio_file, chunk_length_s=10, stride_length_s=(4, 2))
            return output['text']
        elif self.allow_chunking == False:
            output = self.inference.get_transcription(self.model_name, audio_file, self.lm_file)
            return output

