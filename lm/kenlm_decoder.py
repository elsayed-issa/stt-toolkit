#!/usr/bin/python
from typing import Text
from pathlib import Path
import argparse
import os

from clu.speech.build_lm import GetTextData
from clu.speech.build_lm import KenLmDecoder
from clu.speech.build_lm import FixArpaFile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="Generate lm.arpa file for wav2vec2.")

    parser.add_argument("-i", "--input_file", dest="input_file", type=Text, default=None, required=True, help="path to the speech dataset") 
    parser.add_argument("-s", "--save_path", dest="save_path", type=Text, default=None, required=True, help="path to the directory where all kenlm files are saved")
    parser.add_argument("-o", "--output_file", dest="output_file", type=Text, default= "clu-text-kenlm.txt", help="name of the output text file") 

    # FIXME: installation and path to kenlm bin - change to required
    parser.add_argument("-k", "--kenlm_bin", dest="kenlm_bin", type=Text, default= "./clu/speech/kenlm/build/bin/", help="file path to the KENLM binaries lmplz") 
    parser.add_argument("-r","--order", dest="order", type=int, default= 5, help="order of k-grams in the ARPA-file") 
    parser.add_argument("-t", "--text_file", dest="text_file", type=Text, default="clu-text-kenlm.txt", help="name of the text file") 
    parser.add_argument("-p", "--arpa", dest="arpa", type=Text, default="clu-5gram.arpa", help="name of the generated arpa file") 

    parser.add_argument("-f", "--arpa_file", dest="arpa_file", default="clu-5gram.arpa", help="path to the input to the arpa file to be processed for corrections")
    parser.add_argument("-c", "--correct_arpa_file", dest="correct_arpa_file", default='clu-correct-5gram.arpa', help="the final corrected arpa file")
   
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    input_file: Text = args.input_file
    save_path: Text = args.save_path
    output_file: Text = args.output_file

    kenlm_bin: Text = args.kenlm_bin
    order: int = args.order
    text_file: Text = args.text_file
    arpa: Text = args.arpa
    
    arpa_file: Text = args.arpa_file
    correct_arpa_file: Text = args.correct_arpa_file

    # (1) create a directory
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # (2) get the text
    GetTextData(input_file=input_file, save_path=save_path, output_file=os.path.join(save_path, output_file)).clean_and_save()

    # (3) Kenlmize the text
    KenLmDecoder(kenlm_bin=kenlm_bin, order=order, text_file=os.path.join(save_path, text_file), arpa=os.path.join(save_path, arpa)).kenlmize()

    # (4) Fix the arpa file
    FixArpaFile(arpa_file=os.path.join(save_path, arpa_file), correct_arpa_file=os.path.join(save_path, correct_arpa_file)).fix()

