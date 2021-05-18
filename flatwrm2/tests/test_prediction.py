import numpy as np
import flatwrm2
from numpy.testing import assert_almost_equal

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

def test_prediction_validation():
    time,flux,pred_expected = np.genfromtxt(os.path.join(PACKAGEDIR,"kplr003441906-2012032013838_slc.fits.flare.pred"),unpack=True)

    pred, ipred = flatwrm2.prediction([time, flux], batch=64, GPU=0)

    assert(pred.shape == (32543,))
    assert(ipred.shape == (38144, 3))
    assert_almost_equal(pred[176],pred_expected[176],decimal=3)

    flares = flatwrm2.validation( time, flux , ipred,
                            window_size=64,
                            pred_probability_cut=0.5,
                            progressbar=True,
                        )

    assert( flares.shape[0] == 31 )
