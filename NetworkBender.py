
import ddsp
import ddsp.training
import gin
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavutils
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
import json
import librosa
import os
import sounddevice as sd
import sys
from datetime import datetime
from time import sleep
import threading
import mido

class UnitProvider():
    def __init__(self):
        self.unit_list = []
        self.shuffled_list = []
        self.units = 1;

    def get_units(self, s):
        if len(self.unit_list) == 0:
            self.shuffled_list = np.arange(s)
            np.random.shuffle(self.shuffled_list)
        self.unit_list = self.shuffled_list[:int(s * self.units)]
        return self.unit_list

class BendingParam():
    def __init__(self):
        self.t = 0
        #number of vals in block
        self.res = 1000
        self.unit_list = []
        self.lfo = False
        self.ramp = False
        self.scalar = 0
        self.min = 0
        self.max = 1
        self.freq = 1
        self.len = 1
    
    #return 1 block of params
    def get_values(self):
        vals = []
        if self.lfo:
            r = (self.max - self.min) / 2
            vals = np.array([self.step_lfo() for i in range(self.res)])
            vals = vals + (1 + self.min)
            vals = vals * r
        elif self.ramp:
            vals = np.linspace(self.min, self.max, self.len * self.res)[self.t:self.t+self.res]
            self.t = self.t + self.res
        else:
            vals = np.ones(self.res) * self.scalar
        return vals
 
    def step_lfo(self):
        increment = (self.freq / self.res) * (np.pi * 2)
        val = np.sin(self.t)
        self.t = self.t + increment
        return val
        
class BendingTransforms():
    def __init__(self):
        super().__init__()
        self.t = 0
        self.res = 1000
        
    def ablate(self, src, units):
        src = src.numpy()
        src = src.reshape((src.shape[1], src.shape[2]))
        M, N = src.shape
        units = units.get_units(N)
        src[:,units] = 0
        return src.reshape((1, M, N))
    
    def invert(self, src, units):
        src = src.numpy()
        src = src.reshape((src.shape[1], src.shape[2]))
        M, N = src.shape
        units = units.get_units(N)
        src[:,units] = 1 - src[:,units]
        return src.reshape((1, M, N))
    
    def threshold(self, src, thresh, units):
        thresh = thresh.get_values()
        #apply in axis 1 (time)
        thresh = thresh.reshape((thresh.shape[0], 1))
        src = src.numpy()
        one, M, N = src.shape
        src = src.reshape((M, N))
        units = units.get_units(N)
        #print(src[src < t], t, src)
        src[:,units][src[:,units] < thresh] = 0
        src[:,units][src[:,units] >= thresh] = 1
        return src.reshape((1, M, N))
                    
    def step_osc(self, f = 1.0):
        increment = (f / self.res) * (np.pi * 2)
        self.t = self.t + increment
        return np.sin(self.t)
    
    def oscillate(self, src, freq, depth, units):
        src = src.numpy()
        src = src.reshape((src.shape[1], src.shape[2]))
        M, N = src.shape
        f = freq.get_values()
        d = depth.get_values()
        b = np.array([self.step_osc(f[i]) for i in range(0,self.res)]) * d
        #apply in axis 1 (time)
        b = b.reshape(b.shape[0], 1)
        units = units.get_units(N)
        src[:,units] = src[:,units] + b
        return src.reshape((1, M, N))

    def reflect(self, src, r, units):
        alpha = r
        a = np.array([[np.cos(2*alpha), np.sin(2*alpha)],
                      [np.sin(2*alpha), -np.cos(2*alpha)]])
        return self.linear_transformation(src, a)
    
    def rotate(self, src, radians, units):
        alpha = radians
        a = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return self.linear_transformation(src, a)
    
    def linear_transformation(self, src, a):
        src = src.numpy()
        src = src.reshape((src.shape[1], src.shape[2]))
        M, N = src.shape
        points = np.mgrid[0:N, 0:M].reshape((2, M*N))
        new_points = np.linalg.inv(a).dot(points).round().astype(int)
        x, y = new_points.reshape((2, M, N), order='F')
        indices = x + N*y
        wrap = np.take(src, indices, mode='wrap').reshape((1, M, N))
        t = tf.constant(wrap)
        return t

class BendingDecoder(ddsp.training.decoders.RnnFcDecoder):
    def __init__(self):
        super().__init__()
        print("bending init called")

    def init_params(self):
        print("BendingDecoder init_params")
        self.t = {}
        self.t["FC1"] = {}
        self.t["FC2"] = {}
        self.t["GRU"] = {}
    
    def update_transform(self, layer, name, f, a):
        self.t[layer][name] = tf.keras.layers.Lambda(f, arguments = a)
    
    def add_transform(self, layer, name, f, a):
        print("adding transform", layer, name, f, a)
        self.t[layer][name] = tf.keras.layers.Lambda(f, arguments = a)

    def compute_output(self, *inputs):
      # Initial processing.
      #print("BendingDecoder compute_output")
      inputs = [stack(x) for stack, x in zip(self.input_stacks, inputs)]

      # Run an RNN over the latents.
      x = tf.concat(inputs, axis=-1)
      for k,v in self.t["FC1"].items():
          x = v(x)
      x = self.rnn(x)
      for k,v in self.t["GRU"].items():
            x = v(x)
      x = tf.concat(inputs + [x], axis=-1)

      # Final processing.
      x = self.out_stack(x)
      for k,v in self.t["FC2"].items():
            x = v(x)
      return x

class Generator():
    def __init__(self):
        super().__init__()
        self.layers = ["FC1", "GRU", "FC2"]
        self.transforms = {}
        self.buf_length = 16000
        for l in self.layers:
            self.transforms[l] = BendingTransforms()
    
    # setup tensorflow, the feature extractor and the model
    def setup_resynthesis(self, model_dir):
        """
        initialisesm the resynthesis models
        and reset the crepe feature extractor
        """
        #self.setup_tensorflow()
        ddsp.spectral_ops.reset_crepe()
        self.setup_model(model_dir)
        print("setup_resynthesis::resynthesis ready probably")
        self.model.decoder.__class__ = BendingDecoder
        self.model.decoder.init_params()
    
    def setup_tensorflow(self):
        config = tf.compat.v1.ConfigProto()
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)
        print("setup_tensorflow")
        
    def setup_model(self, model_dir):
        gin_file = os.path.join(model_dir, 'operative_config-0.gin')

        if os.path.isfile(gin_file) != True:
            print("setup_model::Gin file not found: ", gin_file)
            return 

         # Parse gin config,
        with gin.unlock_config():
            gin.parse_config_file(gin_file, skip_unknown=True)

        # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
        ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
        ckpt_name = ckpt_files[0].split('.')[0]
        ckpt = os.path.join(model_dir, ckpt_name)

        # Ensure dimensions and sampling rates are equal
        #time_steps_train = gin.query_parameter('DefaultPreprocessor.time_steps')
        time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
        #n_samples_train = gin.query_parameter('Additive.n_samples')
        n_samples_train = gin.query_parameter('Harmonic.n_samples')
        hop_size = int(n_samples_train / time_steps_train)

        time_steps = int(self.buf_length / hop_size)
        required_input_samples = time_steps * hop_size
        print("time steps", time_steps, time_steps_train)
        print("input_samples", required_input_samples, n_samples_train)

        gin_params = [
            'RnnFcDecoder.input_keys = ("f0_scaled", "ld_scaled", "z")',
            'Additive.n_samples = {}'.format(required_input_samples),
            'FilteredNoise.n_samples = {}'.format(required_input_samples),
            'DefaultPreprocessor.time_steps = {}'.format(time_steps),
        ]

        # with gin.unlock_config():
        #     gin.parse_config(gin_params)

        # Set up the model just to predict audio given new conditioning
        self.model = ddsp.training.models.Autoencoder()
        self.model.restore(ckpt) 
        # gin_file = os.path.join(model_dir, 'operative_config-0.gin')
        # gin.parse_config_file(gin_file)
        # self.model = ddsp.training.models.Autoencoder()
        # self.model.restore(model_dir)
    
    def resynth_batch(self, data_dir):
        TRAIN_TFRECORD = data_dir + '/train.tfrecord'
        TRAIN_TFRECORD_FILEPATTERN = TRAIN_TFRECORD + '*'
        data_provider = ddsp.training.data.TFRecordProvider(TRAIN_TFRECORD_FILEPATTERN)
        dataset = data_provider.get_batch(batch_size=1, shuffle=False)

        try:
          batch = next(iter(dataset))
        except OutOfRangeError:
          raise ValueError(
              'TFRecord contains no examples. Please try re-running the pipeline with '
              'different audio file(s).')
        print(batch["f0_hz"].shape)
        audio_gen = self.model(batch, training=False)
        return audio_gen, batch['audio']

    @staticmethod
    def load_audio_data(audio_filename):
        """
        reads all samples from the senf audio_filename
        returns a numpy array of the samples and the sample rate 
        """
        signal, sr=librosa.load(audio_filename, sr=16000, mono = True,)
        print("loaded audio file", len(signal))
        return np.array(signal), sr
        
    def extract_features_and_write_to_file(self, audio_filename):
        """
        looks for a file called audio_filename.csv
        if it does not exist, extracts features
        using the function ddsp.training.metrics.compute_audio_features
        and writes them to that file
        returns the csv filename
        """
        audio_signal, sr = self.load_audio_data(audio_filename)
        feature_filaname = audio_filename + ".csv"
        if not os.path.exists(feature_filaname):
            print("load_features::extracting features from ", audio_filename, ' (slow on CPU!)')
            #audio_features = self.extract_audio_file_features(audio_sig, sr)
            start_time = time.time()
            print('Extracting features (may take a while). Sig length ', len(audio_signal))
            audio_features = ddsp.training.metrics.compute_audio_features(audio_signal, sample_rate=sr)
            print('extract_input_fetures:: Audio features took %.1f seconds' % (time.time() - start_time))
            audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
            stacked = np.stack((audio_features["f0_hz"], audio_features["loudness_db"],audio_features["f0_confidence"]), axis=1)
            df = pd.DataFrame(stacked,columns=["f0_hz","loudness_db","f0_confidence"])
            df.to_csv(feature_filaname)
        else:
            print("features already extracted, found csv") 
        
        return feature_filaname
    
    def write_file(self, output, config = None, normalise = False, sample_rate = 16000):
        complete_output = np.zeros((2, len(output)))
        complete_output[0] = complete_output[1] = output

        print("main:: synthesis ends..." + str(len(output)))

        now = datetime.datetime.now()
        output_root = now.strftime("%Y%m%d%H%M%S")
        output_audio_file = output_root + ".wav"
        output_json_file = output_root + ".json"
        boost_left = boost_right = 1
        if normalise:
          boost_left = self.get_normalise_scalar(complete_output[0])
          boost_right = self.get_normalise_scalar(complete_output[1])
        
        if not config == None:
          output_file = os.path.join(AUDIO_DATA_DIR, output_json_file);
          print("writing config to json", output_file)
          with open(output_file, 'w') as outfile:
              json.dump(config, outfile)

        complete_output[0] = complete_output[0] * boost_left
        complete_output[1] = complete_output[1] * boost_right

        amplitude = np.iinfo(np.int16).max
        complete_output = complete_output * amplitude
        #now rotate it from [[ch1...], [ch2...] to [[c1, c2], [c1, c2] ..]
        complete_output = np.rot90(complete_output, 3) # 3 as 1 is reversed
        #complete_output = np.array([int((x + 1) * 32768) for x in complete_output])
        output_path = os.path.join(AUDIO_DATA_DIR, output_audio_file);
        wavutils.write(output_path, sample_rate, complete_output.astype(np.int16))

        print("main:: wrote result to ", output_file)
        
    def get_normalise_scalar(self, buffer):
        max = 0
        for i in range(len(buffer)):
            if np.abs(buffer[i]) > max:
                max = np.abs(buffer[i])
        scalar = 1/max
        return scalar
    


    def combine_features_and_audio(self, csv_file, audio_file, samplerate = 16000, start = 0.0, end = 1.0):
        """
        creates a basic data structure containing
        features and audio signal 
        assumes the csv_file exists 
        more processing is needed before the data can be fed 
        to the model. That is done by load_and_prepare_features_for_model
        which actually calls me
        """
        #df = pd.read_csv(os.path.join(AUDIO_DATA_DIR,name + ".csv"))
        df = pd.read_csv(csv_file)
        total = np.array(df["f0_hz"]).shape[0]
        start = int(total * start)
        end = int(total * end)
        print("loaded features from {s} to {e}".format(s=start, e=end))
        features = {}
        features["f0_hz"] = np.array(df["f0_hz"])[start:end]
        features["loudness_db"] = np.array(df["loudness_db"])[start:end]
        features["f0_confidence"] = np.array(df["f0_confidence"])[start:end]
        ## note that I don't think we need to add the original
        ## audio signal to the features that are fed to the model
        #features["audio"] = audio_signal[start:end]
        # audio_signal,samplerate = self.load_audio_data(audio_file)
        # total = len(audio_signal)
        # start = int(total * start)
        # end = int(total * end)
        # we do need the sample rate though
        features["sr"] = samplerate
        return features
    
    def load_and_prepare_features_for_model(self, csv_file, audio_file, config, floor = True):    
        """
        gets the input ready for the model
        loads in the features and the signal
        then prepares it in blocks 
        """        
        audio_features = self.combine_features_and_audio(
          #config["features"]["file_name"], 
          csv_file, 
          audio_file, 
          16000, 
          config["features"]["start"],
          config["features"]["end"]
        )
        self.buf_length = config["input_buf_length"]
        self.frames = config["frames"]
        db_boost = config["db_boost"]
        r = np.floor if floor else np.ceil
        steps = r(len(audio_features["f0_hz"]) / self.frames )
        def get_dict(start, af):
            d = {}
            f_start = int(start * self.frames )
            s_start = int(start * self.buf_length)
            d["f0_hz"] = af["f0_hz"][f_start:f_start+self.frames]
            d["loudness_db"] = af["loudness_db"][f_start:f_start+self.frames ] + db_boost
            d["f0_confidence"] = af["f0_confidence"][f_start:f_start+self.frames]
            delta = self.frames - len(d["f0_hz"])
            if delta > 0:
               d["f0_hz"] = np.append(d["f0_hz"], np.zeros(delta))
               d["f0_confidence"] = np.append(d["f0_confidence"], np.zeros(delta))
               d["loudness_db"] = np.append(d["loudness_db"], np.zeros(delta))
            ## note I don't think we need to put the original
            ## audio signal into the features that are fed into the model
            #d["audio"] = [af["audio"][s_start:s_start+self.buf_length]]
            return d

        split = [get_dict(i, audio_features) for i in np.arange(steps)]
        return np.array(split), steps
    
    @staticmethod
    def check_config(config):
        """
        verify the sent config has the correct fields
        uses assert so it will end execution if anything is missing
        """
        want_keys = ["features", "input_buf_length", "frames", "db_boost", "model_dir", "frames"]
        for key in want_keys:
            assert key in config.keys(), "missing config key "+key
            print("check_config::Config has key", key)
        print("check_config::Config looks good")

        
    def get_transform_args(self, f, duration, existing = None):
        """
        Gets the arguments to pass to the transforms lambda layer
        Optionally takes an existing dicionary (if updating not initialising)
        """
        arg = {}
        units = 1;
        #What percentage of units to transform
        if "units" in f.keys():
            units = f["units"]
        if existing == None:
            arg["units"] = UnitProvider()
        else:
            arg["units"] = existing["units"]
        
        arg["units"].units = units
        if "params" in f.keys():
            #Each transform has different named parameters e.g. thresh, freq etc...
            for p in f["params"]:
                if existing == None:
                    arg[p["name"]] = BendingParam()
                else:
                    arg[p["name"]] = existing[p["name"]]
                arg[p["name"]].res = self.frames
                arg[p["name"]].len = int(np.ceil(duration))
                #Inherit the properties from the dict and set on the BendingParam object
                if "args" in p.keys():
                    for k,v in p["args"].items():
                        setattr(arg[p["name"]], k, v)
        return arg
    
    def update_transforms(self, config, duration):
        """
        updates the network bending transforms to the network
        as specified by config
        """
        for l in self.layers:
            #if transforms given for layer l
            #There is one BendingTransforms object for each layer
            if l in config.keys():
                c = config[l]
                #For each function in that layer
                for f in c:
                    name = f["name"]
                    existing_args = self.model.decoder.t[l][name].arguments
                    args = self.get_transform_args(f, duration, existing_args)
                    function = getattr(self.transforms[l], f["function"])
                    self.model.decoder.update_transform(l, name, function, args)

    def add_transforms(self, config, duration):
        """
        adds the network bending transforms to the network
        as specified by config
        """
        for l in self.layers:
        #if transforms given for layer l
        #There is one BendingTransforms object for each layer
            if l in config.keys():
                c = config[l]
                #For each function in that layer
                for f in c:
                    args = self.get_transform_args(f, duration)
                    #For every transform, there is a function and a set of arguments 
                    name = f["name"]
                    function = getattr(self.transforms[l], f["function"])
                    self.model.decoder.add_transform(l, name, function, args)


    def run_features_through_model(self, audio_features):
        """
        runs the sent features through the model one block at a time
        and concatenates the result
        returns an audio signal that is the result
        """
        output = [self.run_feature_block_through_model(i) for i in audio_features]
        faded = []
        output = np.array(output).flatten()
        return output
    
    def run_feature_block_through_model(self, ft):
        """
        runs a single block of features through the model
        returns and audio signal which is the result
        """
        #print("getting next block")
        outputs = self.model(ft, training=False)
        audio = self.model.get_audio_from_outputs(outputs)
        return audio

    
    def play_sine(self):
        start_idx = [0]
        samplerate = 16000
        sd.default.samplerate = samplerate
        sd.default.channels = 1 # only one channel for now!
        def callback(outdata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            t = (start_idx[0] + np.arange(frames)) / samplerate
            t = t.reshape(-1, 1)
            outdata[:] = 0.5 * np.sin(2 * np.pi * 440 * t)
            start_idx[0] += frames

        with sd.OutputStream(channels=1, callback=callback,
                             samplerate=samplerate):
            print('#' * 80)
            sd.sleep(int(60 * 1000))
    
    def start_midi(self,feature_csv_filename, audio_filename, config):
        self.start_realtime(feature_csv_filename, audio_filename, config, True)
    
    def start_realtime(self, feature_csv_filename, audio_filename, config, midi = False):

        sd.default.samplerate = config["sample_rate"]
        sd.default.channels = 1 # only one channel for noe!
        
        # setup the model
        self.setup_resynthesis(config["model_dir"]) 
        # get the features ready
        audio_features, duration = self.load_and_prepare_features_for_model(feature_csv_filename, audio_filename, config)
        # setup the bending transforms
        for l in self.layers:
            self.transforms[l].res = self.frames;
        self.add_transforms(config, duration)
        
        model_buffer_length = config["input_buf_length"]
        config["callback_buffer_length"] = model_buffer_length
        
        audio_ptr = [0]
        output_signal = [self.run_feature_block_through_model(audio_features[audio_ptr[0]])]
        
        if midi:
            def receive_message(message):
                print(message)
                config["FC2"][0]["units"] = message.value/127
                self.update_transforms(config, duration)

            inport = mido.open_input('Akai MPD32 Port 1')
            inport.callback = receive_message

        class GenerateAudioTask: 
            def __init__(self): 
                self._running = True
            def terminate(self): 
                self._running = False
            def run(self, action):
                action(1)

        def generate_audio(ctr):
            sleep(2.5)
            print("generating block")
            audio_ptr[0] = audio_ptr[0] + 1
            output_signal[0] = self.run_feature_block_through_model(audio_features[audio_ptr[0]]) 
            print("done generating block")
       
        c = [0]
        
        def audio_callback(outdata, frames, time, status):
            if status:
                print(status)
            print("block",audio_ptr[0],)
            o = np.reshape(output_signal[0], (-1, 1))
            outdata[:] = o 
            try:
                c[0].terminate()
            except:
                print("no c yet")
            c[0] = GenerateAudioTask() 
            t = threading.Thread(target = c[0].run, args = (generate_audio,))
            t.start()

        with sd.OutputStream(channels=1, samplerate=config["sample_rate"], blocksize=config["callback_buffer_length"], callback=audio_callback):
            for i in np.linspace(0.5,1,100):
                sd.sleep(int(1 * 1000))

    def resynthesize(self, feature_csv_filename, audio_filename, config):
        """
        top level function that does resynthesis
        it prepares the basic models, reads and prepares the features 
        adds transformations then calls
        assumes that the features have already been extracted from
        the input file (audio_filename)
        """
        # setup the model
        self.setup_resynthesis(config["model_dir"]) 
        # get the features ready
        audio_features, duration = self.load_and_prepare_features_for_model(feature_csv_filename, audio_filename, config)
        # setup the bending transforms
        for l in self.layers:
            self.transforms[l].res = self.frames;
        self.add_transforms(config, duration)
        # resynthesize
        output = self.run_features_through_model(audio_features)
        print("DONE")
        return output