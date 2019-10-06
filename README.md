# ts_gnn

### Download the dataset

Download all `.dat` files from [link](https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding/tree/master/Yelp) into `./raw/`

### Complile C++ code for accelerating python and tensorflow

Install ctypes and modify the variable `PYPATH` in `makefile` according to your environment.

Compile the dynamic lib `make`

### Preprocess the raw data

`python preprocess.py`

### Train the model and evaluate the dataset

`python train.py`
