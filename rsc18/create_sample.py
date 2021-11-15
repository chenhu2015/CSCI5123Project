'''
Created on 11.04.2018

@author: malte
'''

import pandas as pd
import numpy as np
import time
from helper import inout
from helper import sample
import math

# data folder
FOLDER = 'data/data_formatted/'
FOLDER_SRC = 'data/data_formatted_50k/'
FOLDER_TEST = '/Users/sophia/Desktop/UMN/Fall 2021/CSCI 5123/Replication Assignment/Data/spotify_million_playlist_dataset_challenge/'
FOLDER_TARGET = 'data/sample_50k'

if __name__ == '__main__':
        
    actions = inout.load_actions(FOLDER, feather=True)
    playlists, artists, tracks = inout.load_meta(FOLDER, feather=True)
    
    inout.ensure_dir( FOLDER_SRC, file=False )
    sample.create_random_training_sample(playlists, artists, tracks, actions, FOLDER_SRC, reduce=0.1)
    inout.convert_feather(FOLDER_SRC)

    actions = inout.load_actions(FOLDER_SRC, feather=True)
    playlists, artists, tracks = inout.load_meta(FOLDER_SRC, feather=True)
    
    challenge_set = inout.load_test(FOLDER_TEST)

    inout.ensure_dir( FOLDER_TARGET+'_similar_2/', file=False )
    sample.create_similar_sample( playlists, actions, challenge_set, FOLDER_TARGET+'_similar_2/', reduce=0.1 )
    #sample.create_random_sample( playlists, actions, challenge_set, FOLDER_TARGET+'_random/', reduce=0.05 )
