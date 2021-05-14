from tkinter import *
from tkinter.ttk import Frame, Button, Entry, Style
DRIVE_DIR = ''
import os
from NetworkBender import Generator
from expand_gpu_memory import expand
import numpy as np

class NetworkBenderFrame(Frame):

    def __init__(self):
        super().__init__()

        self.initUI()
    
    def run(self):
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
        self.update_transforms_from_ui()
        config["transforms"] = self.transforms
        g = Generator()
        g.check_config(config)
        # step 1: write features to CSV file
        feature_csvfile = g.extract_features_and_write_to_file(input_file)

        np.set_printoptions(threshold=np.inf)
        # step 2: do the resynthesis
        audio_gen = g.start_midi(feature_csvfile, input_file, config)
      
    
    def update_transforms_from_ui(self):
        config = []
        
        self.params = {
           "ablate":[],
           "oscillate":["depth","freq"],
           "treshold":["thresh"]
        }
        
        for row in self.all_vars:
            c = {}
            layer = row["layer"].get()
            if not layer == "None" and len(layer) > 0:
                c["layer"] = layer
                c["function"] = row["transform"].get()
                if not c["function"] == "None" and len(c["function"]) > 0:
                    c["units"] = {}
                    c["units"]["value"] = float(row["unit_value"].get())
                    unit_midi = row["unit_midi"].get()
                    if len(unit_midi) > 0:
                        c["units"]["midi"] = {}
                        c["units"]["midi"]["cc"] = int(unit_midi)
                    c["params"] = []
                    for i, param in enumerate(self.params[c["function"]]):
                        p = {}
                        p["name"] = param
                        p["value"] = float(row["param"][i]["param_value"].get())
                        param_midi = row["param"][i]["param_midi"].get()
                        if not unit_midi == "":
                            p["midi"] = {}
                            p["midi"]["cc"] = int(param_midi)
                            p["midi"]["min"] = float(row["param"][i]["param_min"].get())
                            p["midi"]["max"] = float(row["param"][i]["param_max"].get())
                        c["params"].append(p)

                    config.append(c)
        self.transforms = config
        print(self.transforms)
    
    def option_changed(self, *args):
        print(f'You selected: {args}')


    def initUI(self):

        self.master.title("Louis")

        Style().configure("TButton", padding=(0, 5, 0, 5),
            font='serif 10')
        
        NUM_TRANSFORMS = 5
        
        layer_options = [
            "None",
            "FC1",
            "GRU",
            "FC2",
        ]
        
        transform_options = [
            "None",
            "ablate",
            "oscillate",
            "treshold",
        ]
        
        self.all_vars= []

        button = Button(
           self, 
           text="Run", 
           command=self.run
        )
        button.grid(column=0, row = NUM_TRANSFORMS)
        
        for i in range(NUM_TRANSFORMS):
            var_dict = {}
            var_dict["layer"] = StringVar()
            var_dict["transform"] = StringVar()
            var_dict["unit_value"] = StringVar()
            var_dict["unit_value"].set("0.5")
            var_dict["unit_midi"] = StringVar()
            var_dict["param"] = []
            for _ in range(2):
                param_dict = {}
                param_dict["param_value"] = StringVar()
                param_dict["param_value"].set("0.5")
                param_dict["param_midi"] = StringVar()
                param_dict["param_min"] = StringVar()
                param_dict["param_max"] = StringVar()
                var_dict["param"].append(param_dict)
            self.all_vars.append(var_dict)
            
            layer_menu = OptionMenu(self, var_dict["layer"] , *layer_options, command=self.option_changed)
            layer_menu.config(width=10)
            layer_menu.grid(row=i*3, column=1)
            transform_menu = OptionMenu(self, var_dict["transform"] , *transform_options, command=self.option_changed)
            transform_menu.config(width=10)
            transform_menu.grid(row=i*3, column=2)
            
            unit_value = Entry(self, textvariable = var_dict["unit_value"], width=10)
            unit_value.grid(row=i*3, column=3)
            unit_midi = Entry(self, textvariable = var_dict["unit_midi"], width=10)
            unit_midi.grid(row=i*3, column=4)
            
            for j in range(2):
                param_value = Entry(self, textvariable = var_dict["param"][j]["param_value"], width=10)
                param_value.grid(row=(i*3)+(j+1), column=1)
                param_midi = Entry(self, textvariable = var_dict["param"][j]["param_midi"], width=10)
                param_midi.grid(row=(i*3)+(j+1), column=2)
                param_min = Entry(self, textvariable = var_dict["param"][j]["param_min"], width=10)
                param_min.grid(row=(i*3)+(j+1), column=3)
                param_max = Entry(self, textvariable = var_dict["param"][j]["param_max"], width=10)
                param_max.grid(row=(i*3)+(j+1), column=4)
            
 
        self.pack()


def main():

    root = Tk()
    app = NetworkBenderFrame()
    root.mainloop()


if __name__ == '__main__':
    main()