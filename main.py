import soundfile as sf
import argparse
import numpy as np


def readAudioFile(filename):
    data, samplerate = sf.read(filename)
    return data, samplerate

def main():
    print("Inside Main")
    parser = argparse.ArgumentParser()
        ## Required parameters
    parser.add_argument("--audio_filepath",
                        default=None,
                        type=str,
                        required=True,
                        help="The path to a single audio file")
                        
    args = parser.parse_args()
    audio_filepath = args.audio_filepath # C:\Users\19498\OneDrive\Documents\CS229\LA\LA\ASVspoof2019_LA_train\flac\LA_T_1000137.flac
    data, samplerate = readAudioFile(audio_filepath)
    print("Audio File Shape:", data.shape)
    print("samplerate:", samplerate)

if __name__ == "__main__":
    main()