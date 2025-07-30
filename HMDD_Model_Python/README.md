## III. LOW-COST BUT ACCURATE HMDD METHOD ##

### A. Theory in Simple ###

Essentially, this part uses a slightly modified SBCFormer to implement augmented DTM to angle tag mapping. The proposed network is a lightweight ViT-CNN hybrid architecture with fewer model parameters and extremely high inference efficiency, while achieving high accuracy performance at the same time. The code provides two options: Both Matlab version and an improved Python version. 

![HMDD_Network](https://github.com/user-attachments/assets/dacea997-be67-4064-ac07-40b409fd7e5b)

Fig. 2. Schematic of the proposed low-cost accurate HMDD model based on a ViT-CNN hybrid architecture.

#### HMDD_Model_Main.m ####

This script trains and evaluates the HMDD model using the 'Feature_Enhanced_Dataset_OS' dataset, supporting CNN or Transformer architectures, with data preprocessing, model training, and performance evaluation including accuracy and confusion matrix visualization.

**Input:** None (uses predefined `datasetPath` for dataset location; requires `HMDD_Model_Construction_CNN` or `HMDD_Model_Construction_Transformer` functions).

**Output:** None (trains model, saves trained network, and displays accuracy and confusion matrix).

#### HMDD_Model_Main_Improved.py ####

This Python script trains and evaluates a hierarchical ViT-CNN-based model for radar-based human motion direction determination, using data augmentation, label smoothing, and visualization of loss, accuracy, and prediction results on the 'Feature_Enhanced_Dataset_OS' dataset.

**Input:** None (uses predefined `data_dir` for dataset location; requires supporting model scripts).

**Output:** None (saves trained model, plots learning curves, and visualizes prediction results).

### B. Codes Explanation (Folder: HMDD_Model_Matlab, HMDD_Model_Python) ###

#### 1. HMDD_Model_Construction_CNN.m ####

This function constructs a CNN for image classification, featuring an image input layer, multiple convolutional blocks with residual connections, and a classification head, returning an initialized dlnetwork object.

**Input:** Integer `num_classes` (number of classification categories).

**Output:** `HMDD_Model` (dlnetwork object representing the CNN model).

#### 2. HMDD_Model_Construction_Transformer.m ####

This function constructs a Transformer-based neural network for image classification, similar to ViT, with patch embedding, position embedding, multiple Transformer encoder blocks, and a classification head, returning an initialized dlnetwork object.

**Input:** Integer `num_classes` (number of classification categories).

**Output:** `HMDD_Model` (dlnetwork object representing the Transformer model).

#### 3. models.py ####

This Python script defines the model family (XS, S, B, L versions) for image processing, featuring a hierarchical architecture with Stem, SBCFormer Blocks combining local and global feature extraction, and a classification head, tailored for radar-based HMDD tasks.

**Input:** None (defines model architecture with configurable parameters like `img_size`, `num_classes`, `embed_dims`).

**Output:** None (returns model instances for training or evaluation).

### C. Datafiles Explanation (Folder: HMDD_Model_Python\work\model\) ###

#### 1. best_model.pdparams, best_optimizer.pdopt, final_model.pdparams, final_optimizer.pdopt ####

Python version network that stores the model files of the optimal epoch and the last epoch during training.
