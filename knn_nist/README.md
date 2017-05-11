##### Prepare data

- Download preprocess NIST digits from the link https://www.dropbox.com/s/ou5ijdnzndnvt0z/nist.tar.gz?dl=0
- untar the file (`tar -zxvf nist.tar.gz`), and put them into your desired directory `DATA_DIR`
- Change the `data_dir = DATA_DIR` in the `load.py` file

In theory, you can also just use MNIST. But the number of training cases is smaller.



##### Run experiments

`python egan_knn.py`





