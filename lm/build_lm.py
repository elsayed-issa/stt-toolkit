#!/usr/bin/python
from typing import Text, List
import subprocess
import datasets
import re
import os

"""coded based on: 
1- https://huggingface.co/blog/wav2vec2-with-ngram
2- https://github.com/patrickvonplaten/Wav2Vec2_PyCTCDecode
"""


class GetTextData:
    """
    loads audio datasets and extracts the texts from the transcripts file and saves it to `clu-text-kenlm.txt`

    Parameters
    ----------
    input_file:
        path to the speech dataset
    save_path:
        path to a directory that will be used to save the generated text file, the lm.arpa file and the corrected lm.arpa file
    output_file:
        name and path to the text file. Default is `clu-text-kenlm.txt`

    Methods:
    --------
    read_data():
        load a speech dataset and return transcrips as a list of strings.txt
    clean_data(batch):
        static method to clean the text data.
    clean_and_save():
        apply `clean_data(batch)` on the text file and save it to `clu-text-kenlm.txt`.
    
    """
    def __init__(self, input_file: Text, save_path: Text, output_file: Text) -> None:
        self.input_file = input_file
        self.save_path = save_path
        self.output_file = output_file
        self.data = self.read_data()

    def read_data(self) -> List[Text]:
        """
        reads the dataset as follows:
        DatasetDict({
            train: Dataset({
            features: ['audio', 'text', 'path'],
            num_rows: 83
            })
        })
        """
        NOTE: # please according to the dataset used.
        FIXME: # fix for common voice or individual text files. 
        data = datasets.load_dataset(self.input_file)
        text = data['train']['text']
        return text

    @staticmethod
    def clean_data(batch):
        """clean the generated dataset before feeding it to the ASR system."""
        batch = re.sub(r'[\,\?\.\!\-\;\:\"\“\%\‘\”\�\«\»\،\.\:\؟\؛\*\>\<\_\+]', '', batch) + " " # special characters
        batch = re.sub(r'http\S+', '', batch) + " " # links
        batch = re.sub(r'[\[\]\(\)\-\/\{\}]', '', batch) + " " # brackets
        batch = re.sub(r'[\d-]', '', batch) + " " # numbers
        batch = re.sub(r'/[a-zA-Z0-9]+/', '', batch) + " " # numbers
        batch = re.sub(r'\s+', ' ', batch) + "" # extra white space
        return batch.lower().strip()

    def clean_and_save(self):
        """clean the text file and save as `clu-text-kenlm.txt`"""
        cleaned_text = [GetTextData.clean_data(s) for s in self.data]
        with open(self.output_file, "w") as file:
            file.write(" ".join(cleaned_text))


class KenLmDecoder:
    """generate lm.arpa file from a text file

    Parameters
    ----------
    kenlm_bin:
        path to the KENLM binaries 
    order: 
        order of k-grams in ARPA-file
    text_file:
        path to the text file to be used to generate the lm.arpa file
    arpa:
        name and path to the generated arpa file

    Methods
    -------
    kenlmize():
        generates lm.arpa file
    """
    def __init__(self, kenlm_bin: Text, order: Text, text_file: Text, arpa: Text) -> None:
        self.kenlm_bin = kenlm_bin
        self.kenlm_bin = os.path.join(self.kenlm_bin, "lmplz")
        self.order = str(order)
        self.text_file = text_file
        self.arpa = arpa

    def kenlmize(self):
        """generates lm.arpa file from a text file"""
        return subprocess.call([self.kenlm_bin, "--order", self.order, "--text", self.text_file, "--discount_fallback", "--arpa", self.arpa])

class FixArpaFile:
    """fix the generated lm.arpa file 

    Parameters
    ----------
    arpa_file:
        path to the geenrated lm.arpa file
    correct_arpa_file:
        path to the new corrected lm.arpa file

    Methods
    -------
    fix():
        fix the lm.arpa file and generate a new one
    """
    def __init__(self, arpa_file: Text, correct_arpa_file: Text) -> None:
        self.arpa_file = arpa_file
        self.correct_arpa_file = correct_arpa_file

    def fix(self):
        """read the lm.arpa file, fix it and generate a new corrected lm.arpa"""
        with open(self.arpa_file, "r") as read_file, open(self.correct_arpa_file, "w") as write_file:
            has_added_eos = False
            for line in read_file:
                if not has_added_eos and "ngram 1=" in line:
                    count=line.strip().split("=")[-1]
                    write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
                elif not has_added_eos and "<s>" in line:
                    write_file.write(line)
                    write_file.write(line.replace("<s>", "</s>"))
                    has_added_eos = True
                else:
                    write_file.write(line)

