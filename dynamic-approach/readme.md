# EvolveGCN

This folder contains the implementation of the EvolveGCN model for dynamic graph learning, integrated with a Random Forest classifier for enhanced illicit transaction detection in the Elliptic dataset.

## Conda Environment Setup

To replicate the environment used for this project, follow the steps below to install the necessary dependencies using Conda.

### Steps to Install the Conda Environment

1. **Ensure Conda is Installed**: Make sure you have Conda installed on your system. If not, download and install it from the [Anaconda website](https://www.anaconda.com/products/distribution#download-section).

2. **Navigate to the Directory**: Open your terminal or command prompt and navigate to the directory where your `evolvegcn.yaml` file is located.

3. **Create the Conda Environment**: Use the following command to create a new conda environment from the `evolvegcn.yaml` file:
   ```sh
   conda env create -f evolvegcn.yaml

4. **Activate the Environment**: After the environment is created, activate it using:
   ```sh
   conda activate evolvegcn

**Run**
* donwload Elliptic dataset from [kaggle](https://kaggle.com/ellipticco/elliptic-data-set)
* unzip the dataset into a raw directory, such as /home/Elliptic/elliptic_bitcoin_dataset/
* make a new dir to save processed data, such as /home/Elliptic/processed/  
* run train.py by:
```bash
python train.py --raw-dir /home/Elliptic/elliptic_bitcoin_dataset/ --processed-dir /home/Elliptic/processed/
```

**CUDA Compatibility**

Due to the requirements and limitations of the Deep Graph Library (DGL), the .yalm file provided in this repository is only valid for CUDA 11.7 and CUDA 11.8.

For adjustments to other CUDA versions, please refer to the DGL installation guide available at: DGL [Installation Guide](https://www.dgl.ai/pages/start.html)

**Notes**

   - Ensure your system has the compatible CUDA version installed before proceeding with the environment setup.
   - If you encounter any issues or have questions, please feel free to open an issue in this repository.

**Inspiration**

This work has been inspired by the following repository: [EvolveGCN-DGL](https://github.com/maqy1995/EvolveGCN-DGL)
