import os
import sys
import numpy as np
import trax
import trax.layers as tl
from trax.supervised import training
import shutil


PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
from chaosformer.data.maps import HenonMap
from chaosformer.data.generators import TrainDevTest
from chaosformer.core.reconstructor import Reconstructor
from chaosformer.data.pipelines import get_train_stream, create_batch_stream


def training_loop(train_gen, eval_gen, io_size, output_dir = "./.model"):
    '''
    Input:
        TransformerLM (trax.layers.combinators.Serial): The model you are building.
        train_gen (generator): Training stream of data.
        eval_gen (generator): Evaluation stream of data.
        output_dir (str): folder to save your file.
        
    Returns:
        trax.supervised.training.Loop: Training loop.
    '''
    output_dir = os.path.expanduser(output_dir)  # trainer is an object
    lr_schedule = trax.lr.warmup_and_rsqrt_decay(n_warmup_steps=1000, max_value=0.01)

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    train_task = training.TrainTask( 
      labeled_data=train_gen, # None, # The training generator
      # loss_layer=tl.CrossEntropyLoss(), # None, # Loss function 
      loss_layer=tl.L2Loss(), # None, # Loss function 
      optimizer=trax.optimizers.Adam(0.01), # None, # Optimizer (Don't forget to set LR to 0.01)
      lr_schedule=lr_schedule, # None,
      n_steps_per_checkpoint=10
    )

    eval_task = training.EvalTask( 
      labeled_data=eval_gen, # None, # The evaluation generator
      metrics=[tl.L2Loss(), tl.SmoothL1Loss()] # [None, None] # CrossEntropyLoss and Accuracy
    )

    ### END CODE HERE ###

    loop = training.Loop(Reconstructor(state_size=io_size,
                                       d_model=4*16,
                                       d_ff=16,
                                       n_layers=1,
                                       n_heads=2,
                                       max_len=16,
                                       mode='train'),
                         train_task,
                         eval_tasks=[eval_task],
                         output_dir=output_dir)
    
    return loop


def main():
    try:
        shutil.rmtree('./.model')
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    x0 = np.array([0, 0])
    phrase_len = 5
    n_point_dimensionn = 5
    encoding_dimension = x0.shape[0] * n_point_dimensionn

    orbit_size = 995 * 100000
    
    m = HenonMap(a=1.4, b=0.3)
    x0 = np.array([0, 0])
    orbit = m.iterate(x0=x0, n=orbit_size)

    data_generator = TrainDevTest(block_orbit=orbit)
    
    data_generator.generate_blocks(n_point_dimension=n_point_dimensionn)

    data_generator.generate_phrases(phrase_length=phrase_len)

    data_generator.train_dev_test_split_ndx()

    train_iterator = data_generator.train_generator()
    eval_iterator = data_generator.dev_generator()

    train_stream = get_train_stream(train_data_iterator=train_iterator)
    eval_stream = get_train_stream(train_data_iterator=eval_iterator)

    train_batch_stream = create_batch_stream(train_stream)
    eval_batch_stream = create_batch_stream(eval_stream)

    train_input, train_target, train_mask = next(train_batch_stream)

    loop = training_loop(train_gen=train_batch_stream, 
                         eval_gen=eval_batch_stream,
                         io_size = encoding_dimension)
    loop.run(1000)

    


if __name__ == '__main__':
    main()
