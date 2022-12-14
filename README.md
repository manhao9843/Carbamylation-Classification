This repository demonstrates the implementation of the postgraduate dissertation titled *"A Deep Learning Method for Identifying Protein Carbamylation Sites"* written by Hao Man under the supervision of Prof. Martin Cann and Dr. Matteo Degiacomi at Durham University.

### Note: trained models and raw data are available from the [Degiacomi](https://github.com/Degiacomi-Lab) and Cann groups unpon request.
Raw carbamate and non-carbamate sites can be found at [carbamate](https://github.com/manhao9843/AttentionBasedNetwork/blob/main/Data/raw_positive_sites.csv) and [non-carbamate](https://github.com/manhao9843/AttentionBasedNetwork/blob/main/Data/raw_negative_sites.csv).

This work starts with two dataframes documenting the Ground Truth and **PDB or AlphaFold file paths**, which can be obtained from [df_train](https://github.com/manhao9843/AttentionBasedNetwork/blob/main/Data/df_train.csv) and [df_test](https://github.com/manhao9843/AttentionBasedNetwork/blob/main/Data/df_test.csv). The processes can be summarised as follows:
* Data preprocessing (feature extraction) (see [notebook](https://github.com/manhao9843/AttentionBasedNetwork/blob/main/Preparation%20of%20Training%20and%20Test%20Data.ipynb));
* Attention based network training and testing (see [notebook](https://github.com/manhao9843/AttentionBasedNetwork/blob/main/Notebook_AttentionBasedNet.ipynb));
* MLP training and testing (see [notebook](https://github.com/manhao9843/AttentionBasedNetwork/blob/main/Notebook_MLP.ipynb)).

Here shows Attention based network architecture:
![This is an image](https://github.com/manhao9843/AttentionBasedNetwork/blob/main/new%20network.png)
