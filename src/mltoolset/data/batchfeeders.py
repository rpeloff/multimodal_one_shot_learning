"""
TODO(rpeloff) old batch feeder code; need to update/remove/merge this with batch.py
TODO(rpeloff) triplets section adapted from https://github.com/kamperh/tflego/blob/master/tflego/test_siamese.py

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: May 2018
"""


import numpy as np
import scipy.spatial.distance as sci_dist


from .. import _globals


class BatchIterator(object):


    def __init__(self, X, y, batch_size, shuffle_every_epoch=True):
        # Make sure that the data is of type ndarray, so that we do not have to store a duplicate ndarray cast of the data in memory! 
        assert isinstance(X, np.ndarray) or issubclass(type(X), np.ndarray), "Observation data `X` should be an instance or subclass of %s. Found `X` of type %s."  % (np.ndarray, type(X))
        assert isinstance(y, np.ndarray) or issubclass(type(y), np.ndarray), "Observation data `y` should be an instance or subclass of %s. Found `y` of type %s."  % (np.ndarray, type(y))
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle_every_epoch = shuffle_every_epoch
        # Create data indices
        self.indices = np.arange(self.X.shape[0])

    
    def __iter__(self):
        # Shuffle the data indices every epoch
        if self.shuffle_every_epoch:
            shuffle_indices = np.arange(self.indices.shape[0])
            np.random.shuffle(shuffle_indices)
            self.indices = self.indices[shuffle_indices]
        # Calculate the number of batches to determine the stopping iteration, and return the iterator
        self.n_batches = self.indices.shape[0] // self.batch_size
        self.batch_index = 0
        return self

    
    def __next__(self):
        # Check if this is the stopping iteration, and we have iterated over all of the batches
        if self.batch_index == self.n_batches:
            raise StopIteration
        # Get the indices for the next batch
        batch_indices = self.indices[self.batch_index*self.batch_size : (self.batch_index + 1)*self.batch_size]
        # Return the mini-batch
        self.batch_index += 1
        return (
            self.X[batch_indices],
            self.y[batch_indices]
            )


class FewShotTrialIterator(object):


    # TODO: Could add sample_acqui_every_epoch and shuffle every epoch as well?
    def __init__(self, X, y, l_way, k_shot, shuffle=True, z=None):
         # Make sure that the data is of type ndarray, so that we do not have to store a duplicate ndarray cast of the data in memory! 
        assert isinstance(X, np.ndarray) or issubclass(type(X), np.ndarray), "Observation data `X` should be an instance or subclass of %s. Found `X` of type %s."  % (np.ndarray, type(X))
        assert isinstance(y, np.ndarray) or issubclass(type(y), np.ndarray), "Observation data `y` should be an instance or subclass of %s. Found `y` of type %s."  % (np.ndarray, type(y))
        assert (isinstance(y, np.ndarray) or issubclass(type(y), np.ndarray)) if z is not None else True, "Observation data `z` should be an instance or subclass of %s. Found `z` of type %s."  % (np.ndarray, type(z))
        self.X = X
        self.y = y
        self.z = z
        self.l_way = l_way
        self.k_shot = k_shot
        self.shuffle = shuffle
        # Create a list of the few-shot task labels, and check that the chosen `l_way` is not greater than the number of few-shot task labels
        self.task_labels = np.unique(y)
        assert l_way <= self.task_labels.shape[0], "Few-shot task parameter `l_way` greater than maximum number of task labels: %i. Specified value is %i." % (self.task_labels.shape[0], l_way)
        # Sample the few-shot query and acquisition set indices
        self._sample_fewshot_indices()


    def _sample_fewshot_indices(self):
        # Create data indices, where each index is a trial consisting of a query and an acquisition set
        self.query_indices = np.arange(self.X.shape[0])
        self.acqui_indices = []
        # Loop over each trials query index and sample an acquisition set 
        for query_index in self.query_indices:
            curr_acqui_indices = []
            # Sample l-way distinct random labels from the task labels, including the current query label
            l_labels = np.append(
                np.random.choice(
                    self.task_labels[self.task_labels != self.y[query_index]],
                    size=self.l_way - 1,
                    replace=False
                ),
                self.y[query_index]
            )
            # Create a mask of valid data indices for the current acquisition set which excludes the current query index
            valid_mask = self.query_indices != query_index
            # If "originator" data specified, then exclude indices of the originator of the current query from the valid mask
            if self.z is not None:
                valid_mask = valid_mask * (self.z != self.z[query_index])
            # For each of the l-way sampled labels, sample k-shot distinct data indices from the valid data indices that have the same label
            for label in l_labels:
                curr_acqui_indices.append(
                    np.random.choice(
                        self.query_indices[valid_mask * (self.y == label)],
                        size=self.k_shot,
                        replace=False
                    )
                )
            # Append the sampled acquisition set indices to the list of trial acquisition sets
            self.acqui_indices.append(np.array(curr_acqui_indices).flatten())
        self.acqui_indices = np.array(self.acqui_indices)
         # Shuffle the data indices if specified
        if self.shuffle:
            shuffle_indices = np.arange(self.query_indices.shape[0])
            np.random.shuffle(shuffle_indices)
            self.query_indices = self.query_indices[shuffle_indices]
            self.acqui_indices = self.acqui_indices[shuffle_indices]


    def __iter__(self):
        # Set the number of few-shot trials to determine the stopping iteration, and return the iterator
        self.n_trials = self.query_indices.shape[0]
        self.trial_index = 0
        return self
    

    def __next__(self):
        # Check if this is the stopping iteration, and we have iterated over all of the few-shot trials
        if self.trial_index == self.n_trials:
            raise StopIteration
        # Get the indices for the next few-shot trial
        trial_query_index = self.query_indices[self.trial_index]
        trial_acqui_indices = self.acqui_indices[self.trial_index]
        # Return the few-shot trial (along with "originator" data if specified)
        self.trial_index += 1
        if self.z is None:
            return (
                self.X[trial_query_index],
                self.y[trial_query_index],
                self.X[trial_acqui_indices],
                self.y[trial_acqui_indices]
            )
        else:
            return (
                self.X[trial_query_index],
                self.y[trial_query_index],
                self.z[trial_query_index],
                self.X[trial_acqui_indices],
                self.y[trial_acqui_indices],
                self.z[trial_acqui_indices],
            )


class TripletsBatchIterator(object):

    
    def __init__(self, X, y, batch_size, shuffle_every_epoch=True, sample_diff_every_epoch=True, n_same_pairs=int(100e3)):
        # Make sure that the data is of type ndarray, so that we do not have to store a duplicate ndarray cast of the data in memory! 
        assert isinstance(X, np.ndarray) or issubclass(type(X), np.ndarray), "Observation data `X` should be an instance or subclass of %s. Found `X` of type %s."  % (np.ndarray, type(X))
        assert isinstance(y, np.ndarray) or issubclass(type(y), np.ndarray), "Observation data `y` should be an instance or subclass of %s. Found `y` of type %s."  % (np.ndarray, type(y))
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle_every_epoch = shuffle_every_epoch
        self.sample_diff_every_epoch = sample_diff_every_epoch
        self.n_same_pairs = n_same_pairs
        # Sample the "anchor" and "same" indices (i.e. same pairs) from the data once-off during init
        self._sample_same_indices()
        # If not sampling new "different" indices every epoch, or shuffling the same pairs every epoch (which requires sampling new "different" indices), 
        # then sample the "different" indices from the data once-off during init
        if not self.sample_diff_every_epoch or not self.shuffle_every_epoch:
            self._sample_diff_indices()

    
    def _sample_same_indices(self):
        # Generate a matches vector from the label observations
        self.matches_vec = generate_matches_vec(self.y)
        # Sample all possible same pairs, otherwise sample the number of same pairs specified by `n_same_pairs`
        if self.n_same_pairs is None:
            # Convert the matches vector from a condensed vector to a redundant square matrix and get the locations of same pairs in the matrix
            matches_mat = sci_dist.squareform(self.matches_vec)
            match_row_indices, match_col_indices = np.where(matches_mat) # Note: includes non-unique matches, since matches[1,5] same as matches[5,1]
        else:
            # Get the total number of unique same pairs and make sure that the specified `n_same_pairs` is less than this number 
            n_total_pairs = np.where(self.matches_vec == True)[0].shape[0]
            n_pairs = min(self.n_same_pairs, n_total_pairs)
            print("%s: Sampling %i same pairs, with a total number of %i pairs available." % (TripletsBatchIterator, n_pairs, n_total_pairs))
            # Randomly select `n_pairs` number of distinct same pair locations in the matches vector (i.e. indices where matches vector entries are `True`)  
            n_same_samples = np.random.choice(
                np.where(self.matches_vec == True)[0], size=n_pairs, replace=False
            )
            # Create a new matches vector where only the selected `n_same_samples` locations are evaluated as matches, convert it to a matrix, and get the match indices
            n_same_matches_vec = np.zeros(self.matches_vec.shape[0], dtype=np.bool)
            n_same_matches_vec[n_same_samples] = True
            match_row_indices, match_col_indices = np.where(sci_dist.squareform(n_same_matches_vec))
        # Create lists for "anchor" and "same" obaervation indices, and fill them with the matching pairs indices
        self.anch_indices = []
        self.same_indices = []
        for i, j in zip(match_row_indices, match_col_indices):
            self.anch_indices.append(i)
            self.same_indices.append(j)
        self.anch_indices = np.array(self.anch_indices)
        self.same_indices = np.array(self.same_indices)


    def _sample_diff_indices(self):
        # Get the matches matrix and fill the diagonal (i.e. same observations i=j) with `True` for later use when sampling the different pair indices
        matches_mat = sci_dist.squareform(self.matches_vec)
        np.fill_diagonal(matches_mat, True)
        # Initialize an array for "different" example indices that contains a negative one for each "anchor" example index
        self.diff_indices = np.ones(self.anch_indices.shape[0], dtype=_globals.NP_INT) * -1
        # Loop over each label observation (i.e. row in the matches matrix)
        for obs_index in range(matches_mat.shape[0]):
            # Get the locations of the "anchor" indices that match the current observation labels index (i.e. row) 
            obs_anchor_matches = np.where(self.anch_indices == obs_index)[0]
            # For each location that is found, randomly select a "different" index that does not match the current observation labels index (i.e. entry is `False`)
            if obs_anchor_matches.shape[0] > 0:
                self.diff_indices[obs_anchor_matches] = \
                    np.random.choice(
                        np.where(matches_mat[obs_index] == False)[0],
                        size=obs_anchor_matches.shape[0],
                        replace=True
                    )
    

    def __iter__(self):
        """
        Executed at the start of each epoch, when a TripletsBatchIterator instance is used as follows:
        `for ... in <TripletsBatchIterator instance>: ...`
        """
        # Shuffle the same pair indices every epoch, so that they do not contain subsequent indices for the same example data
        if self.shuffle_every_epoch:
            shuffle_indices = np.arange(self.anch_indices.shape[0])
            np.random.shuffle(shuffle_indices)
            self.anch_indices = self.anch_indices[shuffle_indices]
            self.same_indices = self.same_indices[shuffle_indices]
            # Shuffling requires sampling new "different" indices
            self._sample_diff_indices()
        # Just sample new "different" indices every epoch
        elif self.sample_diff_every_epoch:
            self._sample_diff_indices()
        # Calculate the number of batches to determine the stopping iteration, and return the iterator
        self.n_batches = self.anch_indices.shape[0] // self.batch_size
        self.batch_index = 0
        return self
        

    def __next__(self):
        # Check if this is the stopping iteration, and we have iterated over all of the batches
        if self.batch_index == self.n_batches:
            raise StopIteration
        # Get the anchor, same, and different triplet indices for the next batch
        start = self.batch_index*self.batch_size
        end = (self.batch_index + 1)*self.batch_size
        batch_anch_indices = self.anch_indices[start: end]
        batch_same_indices = self.same_indices[start: end]
        batch_diff_indices = self.diff_indices[start: end]
        # Return the triplet data batches
        self.batch_index += 1
        return (
            self.X[batch_anch_indices],
            self.X[batch_same_indices],
            self.X[batch_diff_indices]
        )


def generate_matches_vec(labels):
    """
    Generate a condensed matches vector from the labels `y`. 

    For each observation i and j, where i < j < m, and m is the total number of observations, the boolean match result `y[i] == y[j]` is computed and stored in the matches vector.    
    The matches vector is in the same form as that produced by the scipy.spatial.distance.pdist function.
    """
    # Initialize the condensed matches vector with zeros
    m = len(labels)
    matches_vec = np.zeros(int(m * (m - 1)/2), dtype=np.bool)
    # Loop over each label observation `y[i]`
    label_matches_index = 0
    for i in range(m - 1):
        label = labels[i]
        # Generate an array of matches between `y[i]` and `y[j]`, where i < j < m (m is number of examples)
        label_matches = np.asarray(labels[i+1:]) == label
        # Set the entries in the matches vector from the current label matches index
        matches_vec[label_matches_index : label_matches_index + (m - i - 1)] = label_matches
        # Update the label matches index for the next index, which will start at the current index + (m-i-1)
        label_matches_index += m - i - 1
    return matches_vec
