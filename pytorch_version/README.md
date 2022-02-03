## See the [repository readme](../README.md) for full usage instructions!

### Pytorch implementation
Our repository now supports both **Tensorflow** (at [the main directory](../)) and **Pytorch** (at this directory). 
The two implementations follow a close code and files structure, and share the same interface. 
To switch from the TF to Pytorch, simply enter into `pytorch_version`, and install the [requirements](requirements.txt).

The Pytorch implementation is compatible with models trained through the TF implementation. 
To convert a TF model to Pytorch, run the following:
```python
python loader.py --source=checkpoint-tf.pkl --dest=checkpoint.pkl
```
Compared to the TF version, we removed several options that didn't contribute empricially to performance:
```
--tanh: to add tanh on the generator output
--d-reg: discriminator regularization type [non, gp, r1, r2], (In pytorch we use only r1 -- the default TF regularization)
```

The pytorch version supports the GANformer model, as well as Vanilla and StyleGAN2 baselines. 
The TF version additionally supports k-GAN and SAGAN baselines.

Please feel free to open an issue or [contact](dorarad@cs.stanford.edu) for any questions or suggestions about the new implementation!
