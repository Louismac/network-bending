# Network Bending Neural Vocoders
## Louis McCallun and Matthew Yee-King
Colab host for Network Bending Neural Vocoders demo. Supported by [HDI Network](https://hdi-network.org/) and [MIMIC project](https://hdi-network.org/).

### Paper
Network Bending Neural Vocoders @ NeurIPS 2020, Machine Learning for Creativity and Design Workshop 

We provide one notebook with a model (trained on Whitney Houston) for you to experiment with.

[Open in Colab](https://colab.research.google.com/github/Louismac/network-bending/blob/main/NetworkBending.ipynb)

You can change the tranformations and bending using the config dictionary, instructions below

# Defining Transforms 
## Dictionaries and Arrays in Python
The config that we pass to the program to tell it how we want to make sound is made of two main types of list. 

### Dictionary 
A Dictionary is a type of list where the items are indexed by a key (normally just a word in “brackets”). It is started and ended with curly brackets e.g. { … } and the items in it are separated by commas. 
```
{
   “Item1” : 2,
   “Item2” : ”hat”
}
```
### Array 
An array is a type of list that is just a sequential list of items. It is started and ended by square brackets e.g. [....] and the items in it are separated by commas 
```
[
   2,
   3,
   “hat”
]
```

## Layers 
For each layer (“FC1”, “GRU”, “FC2”), you can define an array of transforms. Each transform is a dictionary. 

```
config[“FC1”] = [
{
   <transform 1>
},
{
   <transform 2>
}
]
```

## Transforms
Each transform dictionary has two key items 
- “function” : the name of the transform
- “units” :  the proportion of units to apply the transform to (between 0->1). 

```
config["FC1"] = [
{
   "function":"invert",
   "units":0.6,
}
]
```

## Parameters
Some transforms have parameters, and these can be constant values, ramped over time or controlled by an lfo. The “params” item in the transform is an array of dictionaries, each telling a specific parameter how to behave.

### Constant value at 0.5
```
config["FC1"] = [
{
   "function":"threshold",
   "units":0.6,
   "params":[
     {
     "name":"thresh",
     "args":{
         "scalar":0.5
       }
     }
   ]
}
]
```
### Ramp from 0->1
```
config["FC1"] = [
{
   "function":"threshold",
   "units":0.6,
   "params":[
     {
     "name":"thresh",
     "args":{
         "ramp":True,
        "min":0,
          "max":1,
       }
     }
   ]
}
]
```

### LFO at 2hz between -1 and 1
```
config["FC1"] = [
{
   "function":"threshold",
   "units":0.6,
   "params":[
     {
     "name":"thresh",
     "args":{
         "lfo":True,
        "min":-1,
          "max":1,
          "freq”:2,  
       }
     }
   ]
}
]
```

## The Transforms 
### Oscillate 
Add an oscillation to the activations in the time dimension. This has two parameters (“freq” and “depth”). Here we use a ramp to gradually increase both of the parameters over time.
```
config["FC1"] = [
{
   "function":"oscillate",
   "units":0.7,
   "params":[
       {"name":"depth",
        "args":{
           "ramp":True,
           "min":0.1,
           "max":0.4,
           }
       },
       {"name":"freq",
        "args":{
           "ramp":True,
           "min":3,
           "max":5,
           }
       }
   ]
}
]
```

### Ablate
Set activations to 0. This has no parameters.
```
config["FC1"] = [
{
   "function":"ablate",
   "units":0.6,
}
]
```

### Invert
1 - activations. This has no parameters.
```
config["FC1"] = [
{
   "function":"invert",
   "units":0.6,
}
]
```

### Shift
Translate all values by a given amount, this kind of works as a shuffle function. There is 1 parameter (“shift_by”) that goes between 0 and 1.
```
config["FC1"] = [
{
   "function":"shift",
   "units":0.6,
   "params":[
     {
     "name":"shift_by",
     "args":{
         "scalar":0.5
       }
     }
   ]
}
]
```

### Threshold
Set all values below the threshold to 0, and all values above the threshold to 1. There is 1 parameter (“thresh”).
``` 
config["FC1"] = [
{
   "function":"threshold",
   "units":0.6,
   "params":[
     {
     "name":"thresh",
     "args":{
         "scalar":0.5
       }
     }
   ]
}
]
```
