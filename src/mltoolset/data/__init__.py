"""Tools for loading, creating, and batching datasets with TensorFlow.

todo(rpeloff) this code was hacked into a mess for the Eloff et al. (2018) paper and needs to be fixed/cleaned.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: Sepetember 2018
"""


# Pull functions from data directory into ml.data namespace:

# Functions from batch.py
from .batch import batch_dataset
from .batch import batch_k_examples_for_p_concepts
from .batch import batch_few_shot_episodes
# from .batch import batch_mulitmodal_few_shot_episodes
from .batch import create_episode_label_set
from .batch import make_train_test_split
from .batch import sample_dataset_triplets
from .batch import sample_triplets

# Functions from load_data.py
from .load_data import load_mnist
from .load_data import load_omniglot
from .load_data import load_flickraudio
from .load_data import load_tidigits
from .load_data import preprocess_images
from .load_data import pad_sequences
# from .load_data import shuffle_data
