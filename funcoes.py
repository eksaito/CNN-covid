# Import
from __future__ import print_function
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os, argparse, pathlib
from eval import eval
from data import process_image_file
from data import apply_augmentation

def getItem(n,
            batch_size,
            input_shape,
            num_channels,
            datasets,
            covid_percent,
            data_dir,
            tipo_dir,
            top_percent,
            augmentation,
            mapping):
    
    batch_x, batch_y = np.zeros((batch_size, *input_shape, num_channels)), np.zeros(batch_size)

    batch_files = datasets[0][n * batch_size:(n + 1) * batch_size]
    
    # upsample covid cases
    covid_size = max(int(len(batch_files) * covid_percent), 1)
    covid_inds = np.random.choice(np.arange(len(batch_files)), size=covid_size, replace=False)
    covid_files = np.random.choice(datasets[1], size=covid_size, replace=False)
    for i in range(covid_size):
        batch_files[covid_inds[i]] = covid_files[i]

    for i in range(len(batch_files)):
        sample = batch_files[i].split()

        folder = tipo_dir

        x = process_image_file(os.path.join(data_dir, folder, sample[1]),
                               top_percent,
                               input_shape[0])

        if augmentation:
            x = apply_augmentation(x)

        x = x.astype('float32') / 255.0
        y = mapping[sample[2]]

        batch_x[i] = x
        batch_y[i] = y
        
    return batch_x, batch_y

def gen_batch(data_dir,
              tipo_dir,
              csv_file,
              batch_size,
              input_shape,
              num_channels,
              covid_percent,
              top_percent):
    
    with open(csv_file, 'r') as fr:
        dataset = fr.readlines()
        
    mapping={
                'normal': 0,
                'pneumonia': 1,
                'COVID-19': 2
            }
    n = 0
    
    datasets = {'normal': [], 'pneumonia': [], 'COVID-19': []}
    for l in dataset:
        datasets[l.split()[2]].append(l)
        
    datasets = [
        datasets['normal'] + datasets['pneumonia'],
        datasets['COVID-19'],
    ]
    
    N = int(np.floor(len(datasets[0]) / float(batch_size)))
    print(N)
    
    if tipo_dir != 'test':
        augmentation = True
    else:
        augmentation = False
            
    while True:
        batch_x, batch_y = getItem(n,
            batch_size,
            input_shape,
            num_channels,
            datasets,
            covid_percent,
            data_dir,
            tipo_dir,
            top_percent,
            augmentation,
            mapping)
    
        n = n + 1
        if n >= N:
            n = 0
            for v in datasets:
                np.random.shuffle(v)
                
        yield(batch_x, batch_y)