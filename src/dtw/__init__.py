"""Dynamic Time Warping implemented in Cython by Herman Kamper.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: Sepetember 2018
"""


# Pull in functions from speech DTW cython directory:
from .speech_dtw import _dtw as speech_dtw
