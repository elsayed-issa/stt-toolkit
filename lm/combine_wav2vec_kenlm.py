#!/usr/bin/env python3
from typing import Text
from transformers import AutoProcessor
from transformers import Wav2Vec2ProcessorWithLM
from pyctcdecode import build_ctcdecoder
import subprocess
import argparse
import os


"""This script combine both a pretrained wav2vec model and an arpa language models. This script is based on the following tutorial
https://huggingface.co/blog/wav2vec2-with-ngram
"""

class Wav2Vec2Kenlm:
    def __init__(self, model_name: Text, arpa_file: Text, out_file: Text, kenlm: Text) -> None:
        self.model_name = model_name
        self.arpa_file = arpa_file
        self.out_file = out_file
        self.kenlm = kenlm
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.vocab_dict = self.processor.tokenizer.get_vocab()
        self.sorted_vocab_dict = {k.lower(): v for k, v in sorted(self.vocab_dict.items(), key=lambda item: item[1])}
        self.decoder = build_ctcdecoder(labels=list(self.sorted_vocab_dict.keys()), kenlm_model_path=self.arpa_file,)

    def build_model(self):
        processor_with_lm = Wav2Vec2ProcessorWithLM(feature_extractor=self.processor.feature_extractor,tokenizer=self.processor.tokenizer,decoder=self.decoder)
        processor_with_lm.save_pretrained(self.out_file)
        return processor_with_lm

    def convert_to_binary(self):
        return subprocess.call([self.kenlm, os.path.join(self.out_file, "language_model/clu-correct-5gram.arpa"), os.path.join(self.out_file, "language_model/clu-5gram.bin")])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="Generate lm.arpa file for wav2vec2.")

    parser.add_argument("-m", "--model_name", dest="model_name", type=Text, default=None, required=True, help="path to the speech dataset") 
    parser.add_argument("-a", "--arpa_file", dest="arpa_file", type=Text, default=None, required=True, help="path to the directory where all kenlm files are saved")
    parser.add_argument("-o", "--out_file", dest="out_file", type=Text, default=None, help="name of the output text file") 
    parser.add_argument("-k", "--kenlm", dest="kenlm", type=Text, default="./clu/speech/kenlm/build/bin/build_binary", help="path to the speech dataset") 
    

    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    model_name: Text = args.model_name
    arpa_file: Text = args.arpa_file
    out_file: Text = args.out_file
    kenlm: Text = args.kenlm

    x = Wav2Vec2Kenlm(model_name=model_name, arpa_file=arpa_file, out_file=out_file, kenlm=kenlm)
    x.build_model()
    x.convert_to_binary()
    
    




