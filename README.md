This repository demonstrates the implementation of the postgraduate dissertation titled *"A Deep Learning Method for Identifying Protein Carbamylation Sites"* written by Hao Man under the supervision of Prof. Martin Cann and Dr. Matteo Degiacomi at Durham University.

This work starts with a dataframe documenting the Ground Truth, which can be obtained from [df_train](.../Data/df_train.csv) and [df_test](.../Data/df_test.csv). The processes can be summarised as follows:
* Data preprocessing (feature extraction) (see [notebook](https://github.com/manhao9843/AttentionBasedNetwork/blob/main/Preparation%20of%20Training%20and%20Test%20Data.ipynb));
* Attention based network training and testing (see [notebook](https://github.com/manhao9843/AttentionBasedNetwork/blob/main/Notebook_AttentionBasedNet.ipynb));
* MLP training and testing (see [notebook](https://github.com/manhao9843/AttentionBasedNetwork/blob/main/Notebook_MLP.ipynb)).

Here shows Attention based network architecture:
![This is an image](https://github.com/manhao9843/AttentionBasedNetwork/blob/main/Network.png)
