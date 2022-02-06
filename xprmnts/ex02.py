import os
import sys
import numpy as np
import trax
import trax.layers as tl
from trax.supervised import training

import textwrap
wrapper = textwrap.TextWrapper(width=70)

import trax
from trax import layers as tl
from trax.fastmath import numpy as jnp

# to print the entire np array
np.set_printoptions(threshold=sys.maxsize)


def main():
    # Importing CNN/DailyMail articles dataset
    train_stream_fn = trax.data.TFDS('cnn_dailymail',
                                    data_dir='.data/',
                                    keys=('article', 'highlights'),
                                    train=True)

    # This should be much faster as the data is downloaded already.
    eval_stream_fn = trax.data.TFDS('cnn_dailymail',
                                    data_dir='.data/',
                                    keys=('article', 'highlights'),
                                    train=False)

    def tokenize(input_str, EOS=1):
        """Input str to features dict, ready for inference"""
    
        # Use the trax.data.tokenize method. It takes streams and returns streams,
        # we get around it by making a 1-element stream with `iter`.
        inputs =  next(trax.data.tokenize(iter([input_str]),
                                        vocab_dir='vocab_dir/',
                                        vocab_file='summarize32k.subword.subwords'))
        
        # Mark the end of the sentence with EOS
        return list(inputs) + [EOS]

    def detokenize(integers):
        """List of ints to str"""
    
        s = trax.data.detokenize(integers,
                                vocab_dir='vocab_dir/',
                                vocab_file='summarize32k.subword.subwords')
        
        return wrapper.fill(s)
    
    # Special tokens
    SEP = 0 # Padding or separator token
    EOS = 1 # End of sentence token

    # Concatenate tokenized inputs and targets using 0 as separator.
    def preprocess(stream):
        for (article, summary) in stream:
            joint = np.array(list(article) + [EOS, SEP] + list(summary) + [EOS])
            mask = [0] * (len(list(article)) + 2) + [1] * (len(list(summary)) + 1) # Accounting for EOS and SEP
            yield joint, joint, np.array(mask)

    # You can combine a few data preprocessing steps into a pipeline like this.
    input_pipeline = trax.data.Serial(
        # Tokenizes
        trax.data.Tokenize(vocab_dir='.vocab_dir/',
                        vocab_file='summarize32k.subword.subwords'),
        # Uses function defined above
        preprocess,
        # Filters out examples longer than 2048
        trax.data.FilterByLength(2048)
    )

    # Apply preprocessing to data streams.
    train_stream = input_pipeline(train_stream_fn())
    eval_stream = input_pipeline(eval_stream_fn())

    train_input, train_target, train_mask = next(train_stream)

    assert sum((train_input - train_target)**2) == 0  # They are the same in Language Model (LM).

    emb_l = tl.Embedding(vocab_size=33300, d_feature=10)

    sdtype = trax.shapes.ShapeDtype(shape=train_input.shape)
    emb_l.init(input_signature=sdtype)

    te = emb_l(x=train_input)

    te = emb_l(x=train_target)

    te = None


if __name__ == '__main__':
    main()
