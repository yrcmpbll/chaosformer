from cmath import phase
import trax
import numpy as np


# Concatenate tokenized inputs and targets using 0 as separator.
def preprocess(stream):

    for phrase in stream:

        # joint = np.array(phrase)
        # joint = np.stack(phrase).transpose()
        joint = np.stack(phrase)
        # mask = [1] * (len(phrase) + 1)
        # mask = [0] * (len(phrase))
        mask = np.ones_like(joint)

        # joint = np.array(list(article) + [EOS, SEP] + list(summary) + [EOS])
        # mask = [0] * (len(list(article)) + 2) + [1] * (len(list(summary)) + 1) # Accounting for EOS and SEP
        
        yield joint, joint, np.array(mask)


def get_train_stream(train_data_iterator):

    # You can combine a few data preprocessing steps into a pipeline like this.
    input_pipeline = trax.data.Serial(
        
        # Uses function defined above
        preprocess,

        # Filters out examples longer than 2048
        trax.data.FilterByLength(2048)
    )

    # Apply preprocessing to data streams.
    train_stream = input_pipeline(train_data_iterator)

    return train_stream


# eval_stream = input_pipeline(eval_stream_fn())

# train_input, train_target, train_mask = next(train_stream)

# assert sum((train_input - train_target)**2) == 0  # They are the same in Language Model (LM).

def create_batch_stream(stream):

    # boundaries =  [128, 256,  512, 1024]
    boundaries = [8]
    # batch_sizes = [16,    8,    4,    2,   1]
    batch_sizes = [16, 8, 4, 2, 1]

    # Create the streams.
    batch_stream = trax.data.BucketByLength(boundaries, batch_sizes)(stream)

    return batch_stream