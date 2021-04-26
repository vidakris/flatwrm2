# FLATW'RM2

FLATW'RM2 is a deep learning code that was developed to detect flares in light curves obtained by space observatories (like Kepler or TESS).

The code was developed using <a href="https://keras.io">Keras</a>: a deep learning API built on Tensorflow.

The code depends on the following packages:
* tensorflow
* numpy
* scipy
* numba
* matplotlib (for testing)

Additionally, for `ocelot-gym.py`, if you want to create your own artificial training files for any reason, you will need `pyMacula` and the analog flare model for injection from Davenport (2016ApJ...829...23D) in the `aflare.py` file.

To use the network with the current weight file and a batch size of 16 on GPU 2, use:
`flatwrm2 -m LSTM_weights_keplerSC.h5 -b 16 --gpu=2 light_curve.dat`
For general help on usage, try:
`flatwrm2 --help`.

Alternatively, you can use `flatwrm2` as a module from jupyter-notebook, check the `example.ipynb` notebook in the examples.


For retraining the network, you can use the `ocelot-gym` code to generate artificial spotted, flaring light curves, but these shouldn't be the only training data for the network, make sure there is enough real light curves with the flares flagged in it.

Once you have your training data, you can use the `flatwrm2-training.ipynb` notebook to train your network, either from scratch, or starting from one of the weight files. The runtime of the training is in the order of 5-10 hours on a single GPU.

<!--<img src="flatwrm-mark2.png" width="250">-->
<p align="center">
  <br><br>
<img src="./figures/flatwrm-mark2.png" width="300">
</p>
