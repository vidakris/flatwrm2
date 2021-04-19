# FLATW'RM2 - coming soon!

FLATW'RM2 is a deep learning code that was developed to detect flares in light curves obtained by space observatories (like Kepler or TESS).

The code was developed using <a href="https://keras.io">Keras</a>: a deep learning API built on Tensorflow.

The code depends on the following packages:
* tensorflow
* numpy
* scipy
* matplotlib (for testing)

To use the network with the current weight file and a batch size of 16 on GPU 2, use:
`./flatwrm2-predict.py -m LSTM-fold_all-mixedtrain0.h5 -b 16 --gpu=2 light_curve.dat`
multithread processing is partly implemented, but didn't seem to be effective, so it's commented out at the end, and `--number=2` or larger doesn't do anything, just turns off verbosity. For general help on usage, try:
`./flatwrm2-predict.py --help`.


For retraining the network, you can use the `ocelot-gym` code to generate artificial spotted, flaring light curves, but these shouldn't be the only training data for the network, make sure there is enough real light curves with the flares flagged in it.

Once you have your training data, you can use the `flatwrm2-training.ipynb` notebook to train your network, either from scratch, or starting from one of the weight files. The runtime of the training is in the order of 5-10 hours on a single GPU. 

<!--<img src="flatwrm-mark2.png" width="250">-->
<p align="center">
  <br><br>
<img src="flatwrm-mark2.png" width="300">
</p>
