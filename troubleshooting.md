# Tensorflow
* **libcublas image not found**: change tensorflow-gpu for tensorflow in
  requirements.txt. Uninstall tensorflow and re run `pip install -r requirements.txt`
* **python must be installed as a framework**: add a matplotlibrc file in
  .matplotlib with the line `backend: TkAgg` [check this thread](https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python)
* **No such file or directory: 'data/networks/ggcnn_rss/_val_input.npy'**: This
  happens when trying to evaluate a network that you haven't trained yourself.
  *val_input.npy* contains the validation images used when training.

# Original GG-CNN
* **generate_dataset.py**: creates a hdf5 file in data/datasets with
  augmented/preprocessed data propely split into training/testing sets from raw dataset sources
