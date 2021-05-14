DRIVE_DIR = ''
import os
from NetworkBender import Generator
from expand_gpu_memory import expand
import numpy as np

model_name = "Flute2021New"
DRIVE_DIR = '.'
if DRIVE_DIR:
  MODEL_DIR = os.path.join(DRIVE_DIR, 'Models/' + model_name)
  AUDIO_DATA_DIR = os.path.join(DRIVE_DIR, 'audio_data')


expand()

##See Instructions (https://github.com/Louismac/network-bending/blob/main/README.md)
samplerate = 16000
input_file = AUDIO_DATA_DIR + "/monk-48.wav"
config = {}
config["model_dir"] = MODEL_DIR
config["sample_rate"] = samplerate
#pick how much of input file to do (0->1)
config["features"] = {"file_name":input_file, "start":0, "end":1}
#add boost to loudness feature of input
config["db_boost"] = 10
#4 secs at 16000
config["input_buf_length"] = 4 * samplerate
config["frames"] = 1000
#transforms for first layer
# config["FC2"] = [
#     {
#        "name":"transform1",
#        "function":"ablate",
#        "units":0.5,
#     }
# ]

config["transforms"] = [
    {
        "layer":"FC1",
        "function":"oscillate",
        "units":
        {                
            "value":0.7,
            "midi":
            {
                "cc":23
            }
        },
        "params":[
            {
                "name":"depth",
                "value":1,
                "midi":
                {
                    "min":0,
                    "max":1,
                    "cc":24
                }
            },
            {
                "name":"freq",
                "value":5,
                "midi":
                {
                    "min":3,
                    "max":20,
                    "cc":25
                }
            },
        ]
    }
]

g = Generator()
g.check_config(config)
# step 1: write features to CSV file
feature_csvfile = g.extract_features_and_write_to_file(input_file)

np.set_printoptions(threshold=np.inf)
# step 2: do the resynthesis
audio_gen = g.start_midi(feature_csvfile, input_file, config)
#g.play_sine()
