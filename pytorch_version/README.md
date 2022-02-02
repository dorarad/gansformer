## See the [repository readme](../README.md) for full usage instructions!

### Pytorch implementation
Our repository now supports both **Tensorflow** (at [the main directory](../)) and **Pytorch** (at this directory). 
The two implementations follow a close code and files structure, and share the same interface. 
To switch from the TF to Pytorch, simply enter into `pytorch_version`, and install the [requirements](requirements.txt).

The Pytorch implementation is compatible with models trained through the TF implementation. 
To convert a TF model to Pytorch, run the following:
```python
python loader.py --source=checkpoint.pkl --dest=checkpoint.pkl
```

Please feel free to open an issue or [contact](dorarad@cs.stanford.edu) for any questions or suggestions about the new implementation!
