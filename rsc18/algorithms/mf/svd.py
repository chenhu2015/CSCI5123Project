import numpy as np
import pandas as pd
from scipy import sparse
import time
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import KFold

class My_SVD:
    def __init__(self, n_factors = 100, n_epochs = 10, lr = 0.01, session_key = 'playlist_id', item_key = 'track_id', artist_key = 'artist_id'):
        self.factors = n_factors
        self.epochs = n_epochs
        self.lr = lr
        self.session_key = session_key
        self.item_key = item_key
        self.artist_key = artist_key

    def train(self, train, test=None):
        data = train['actions']
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        self.itemidmap2 = pd.Series(index=np.arange(self.n_items), data=itemids)

        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        self.useridmap = pd.Series(data=np.arange(self.n_sessions), index=sessionids)
        
        artistids = data[self.artist_key].unique()
        self.n_artists = len(artistids)
        self.ratingidmap = pd.Series(data=np.arange(self.n_artists), index=artistids)
        
        tstart = time.time()
        
        data = pd.merge(data, pd.DataFrame({self.item_key:self.itemidmap.index, 'ItemIdx':self.itemidmap[self.itemidmap.index].values}), on=self.item_key, how='inner')
        data = pd.merge(data, pd.DataFrame({self.session_key:self.useridmap.index, 'SessionIdx':self.useridmap[self.useridmap.index].values}), on=self.session_key, how='inner')
        data['Rating'] = 1

        print( 'add index in {}'.format( (time.time() - tstart) ) )
        self.model = SVD(n_factors=self.factors, n_epochs=self.epochs, lr_all=self.lr)

        reader = Reader(rating_scale=(0, 1))
        dataset = Dataset.load_from_df(data[['SessionIdx', 'ItemIdx', 'Rating']], reader)

        trainset = dataset.build_full_trainset()

        self.model.fit(trainset)
    
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
        
        items = tracks if tracks is not None else []

        if len(items) == 0:
            res_dict = {}
            res_dict['track_id'] = []
            res_dict['confidence'] = []
            return pd.DataFrame.from_dict(res_dict)
        
        itemidxs = self.itemidmap[items]
        # Haven't considered idf here, could use as future improvement
        uF = self.model.qi[itemidxs].mean(axis=0).copy()
        conf = np.dot( self.model.qi, uF )
        res_dict = {}
        res_dict['track_id'] =  self.itemidmap.index
        res_dict['confidence'] = conf

        res = pd.DataFrame.from_dict(res_dict)
        res = res[ ~np.in1d( res.track_id, tracks ) ]
        res.sort_values( 'confidence', ascending=False, inplace=True )

        return res.head(500)
        
