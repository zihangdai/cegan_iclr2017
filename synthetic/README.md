#### Data preparation 

Firstly, run `python synthetic_data.py` to create the synthetic datasets. Data will be stored in the groundtruth folder.



#### Run the experiment using KNN entropy approximation. 

Then, configurate the dataset in the `def egan_state(...)` function within the file  `egan_knn_state.py`. 

Finally, run `python egan_knn.py` to start the training.







