from math import floor
from chaosformer.data.orbits import make_chunks
import random as rnd


class TrainDevTest:

    def __init__(self, block_orbit) -> None:
        self.block_orbit = block_orbit
    
    def generate_blocks(self, n_point_dimension):
        self.block_orbit.build_blocked_orbit(n_point_dimension=n_point_dimension)
    
    def generate_phrases(self, phrase_length):
        self.phrases = [ph for ph in make_chunks(l=self.block_orbit.blocked_orbit, n=phrase_length) if len(ph) == phrase_length]
        self.n_phrases = len(list(self.phrases))
    
    def train_dev_test_split_ndx(self, train_frac=.8):
        rest_frac = 1 - train_frac

        self.start_train_ndx = 0
        self.end_train_ndx = floor(train_frac * self.n_phrases)
        
        self.start_dev_ndx = self.end_train_ndx + 1
        self.end_dev_ndx = floor((train_frac + rest_frac/2) * self.n_phrases)

        self.start_test_nxd = self.end_dev_ndx + 1
        self.end_test_nxd = self.n_phrases - 1
    

    def train_phrases(self):
        return [phrase for phrase in self.phrases[self.start_train_ndx : self.end_train_ndx]]

    def dev_phrases(self):
        return [phrase for phrase in self.phrases[self.start_dev_ndx : self.end_dev_ndx + 1]]
    
    def test_phrases(self):
        return [phrase for phrase in self.phrases[self.start_test_nxd : self.end_test_nxd + 1]]
    
    def train_generator(self, randomize=True):
        t_phrases = self.train_phrases()

        idx = 0
        len_phrases = len(t_phrases)
        phrase_indexes = [*range(len_phrases)]
        
        if randomize:
            rnd.shuffle(phrase_indexes)

        while True:
            if idx >= len_phrases:
                # if idx is greater than or equal to len_q, set idx accordingly 
                # (Hint: look at the instructions above)
                idx = 0 # None
                # shuffle to get random batches if shuffle is set to True
                if randomize:
                    rnd.shuffle(phrase_indexes)
            idx += 1
            yield t_phrases[phrase_indexes[idx]]


    
    def dev_generator(self):
        for phrase in self.dev_phrases():
            yield phrase
    
    def test_generator(self):
        for phrase in self.test_phrases():
            yield phrase