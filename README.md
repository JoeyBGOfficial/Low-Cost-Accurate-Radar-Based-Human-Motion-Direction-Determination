# Low-Cost-Accurate-Radar-Based-Human-Motion-Direction-Determination
## I. Introduction ##

### Write Sth. Upfront: ###

So this is the second project that I am publishing on a preprint platform, which is not planning to submit to a journal or conference, but is open source. The main purpose of this work is to provide an angle prior for radar-based human gait recognition tasks. At the same time, we hope to explore a relatively accurate and low-cost solution.

### Basic Information: ###

This repository is the open source code for my latest work: "Exploration of Low-Cost but Accurate Radar-Based Human Motion Direction Determination", submitted to arXiv.

$\textcolor{red}{\textbf{Full Version of Paper}}$ can be downloaded at: https://smallpdf.com/file#s=71472927-4d9e-4cb6-a1b2-76bc9b5cf0d4.

**ArXiv Simplified Version of Paper** can be downloaded at:

**My Email:** JoeyBG@126.com;

**Abstract:** This work is completed on a whim after discussions with my junior colleague. The motion direction angle affects the micro-Doppler spectrum width, thus determining the human motion direction can provide important prior information for downstream tasks such as gait recognition. However, Doppler-Time map (DTM)-based methods still have room for improvement in achieving feature augmentation and motion determination simultaneously. In response, a low-cost but accurate radar-based human motion direction determination (HMDD) method is explored in this paper. In detail, the radar-based human gait DTMs are first generated, and then the feature augmentation is achieved using feature linking model. Subsequently, the HMDD is implemented through a lightweight and fast Vision Transformer-Convolutional Neural Network hybrid model structure. The effectiveness of the proposed method is verified through open-source dataset.

**Corresponding Papers:**

[1] W. Gao, “Exploration of Low-Cost but Accurate Radar-Based Human Motion Direction Determination,” arXiv (Cornell University), August. 2025, Link: .

### Important!!! ###

**The data used in this paper's experiments were taken entirely from: L. Du, X. Chen, Y. Shi, S. Xue and M. Xie, “MMRGait-1.0: A radar time-frequency spectrogram dataset for gait recognition under multi-view and multi-wearing conditions,” J. Radars, vol. 12, no. 4, pp. 892-905, 2023.**

**First, download the complete open source dataset and place all subfolders from "001" to "121" in the "Open_Source_Dataset\Original_MAT_Dataset_OS\" path. Next, run the code. After my debugging, if the software version and third-party libraries are installed correctly, there should be no error messages.**

## II. MICRO-DOPPLER SIGNATURE AUGMENTATION ##

### A. Theory in Simple ###

The open-source dataset already provides DTM images stored in ".mat" format, so this paper does not need to implement signal preprocessing in code. The proposed method mainly uses improved FLM to achieve micro-Doppler signature extraction and enhancement. 

![FLM](https://github.com/user-attachments/assets/2d1d368c-bc51-4a76-8d5b-e89028bbb1cb)

Fig. 1. Schematic of the proposed FLM-based feature augmentation method.

#### Dataset_Reconstruction_OS.m ####

This script organizes .mat files from a dataset by traversing 121 subfolders and their secondary subfolders, sorting files into one of eight target folders based on a three-digit direction code in "Open_Source_Dataset\Direction_Dataset_OS\".

**Input:** None (uses predefined `main_path` and `target_path` for source and destination directories).

**Output:** None (copies .mat files to target folders: `000`, `030`, `045`, `060`, `090`, `300`, `315`, `330`).

#### Feature_Augmentation_OS.m ####

This script processes .mat files containing 'tfmap' variables from eight subfolders, applies feature augmentation using the FLM_Processing function, and saves the enhanced images as .png files with a 'jet' colormap in corresponding output subfolders "Open_Source_Dataset\Feature_Enhanced_Dataset_OS\".

**Input:** None (uses predefined `root_dir` for input dataset and `output_root` for output directory; requires `tfmap` in .mat files).

**Output:** None (saves augmented images as .png files in subfolders `000`, `030`, `045`, `060`, `090`, `300`, `315`, `330`).

### B. Codes Explanation (Folder: FLM) ###

#### 1. FLM_Processing.m ####

This function enhances an input image using the FLM after adaptive histogram equalization and thresholding, resizing the output to a specified resolution.

**Input:** 2D/3D matrix `tfmap` (grayscale or RGB image); Float `Cutting_Threshold` for low-intensity pixel cutoff; Integer `Estimation_Resolution` for output image size.

**Output:** 2D matrix `FLM_Enhancement` (enhanced image).

#### 2. v2rgb.m ####

This function combines a processed V channel with the original image’s H and S channels to produce an enhanced RGB image, or returns the V channel directly for grayscale inputs.

**Input:** 2D/3D matrix `I` (grayscale or RGB image); 2D matrix `V` (processed V channel).

**Output:** 2D/3D matrix `Io` (enhanced RGB or grayscale image).

#### 3. rgb2v.m ####

This function extracts the V component from an RGB image in HSV color space, or returns the input unchanged if it is grayscale.

**Input:** 2D/3D matrix `I` (grayscale or RGB image).

**Output:** 2D matrix `Iv` (V channel or original grayscale image).

#### 4. QEvaluation.m ####

This function assesses the quality of an enhanced grayscale image by computing Local Contrast (LC), Spatial Frequency (SF), and Mean Gradient (MG).

**Input:** 2D matrix `I` (grayscale image).

**Output:** Scalars `lc` (local contrast), `SF` (spatial frequency), `MG` (mean gradient).

#### 5. GrayStretch.m ####

This function enhances contrast in a grayscale image by stretching pixel intensities to the full 0-255 range based on histogram bounds.

**Input:** 2D matrix `I` (grayscale image); Float `Per` (percentage of histogram to stretch).

**Output:** 2D matrix `GS` (stretched grayscale image).

#### 6. FLM.m ####

This function implements the Feature-Linking Model (FLM) to enhance grayscale images through iterative processing, followed by gray-level stretching.

**Input:** 2D matrix `I` (grayscale image).

**Output:** 2D matrix `Rep1gs` (enhanced grayscale image).

### C. Datafiles Explanation (Folder: Open_Source_Dataset\Feature_Enhanced_Dataset_OS_64.zip) ###

#### 1. Feature_Enhanced_Dataset_OS_64.zip ####

Here we provide a set of feature-enhanced data in ".zip" file stored as $64 \times 64$ RGB ".png" files.

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

## IV. Visualization ##

### A. Theory in Simple ###

The experiment visualizeseight different angles of human motion using the example DTMs and their augmented images.

![Visualizations](https://github.com/user-attachments/assets/08cf1df3-263e-41df-8cf2-a83c1e7d0d13)

Fig. 3. Visualization of DTMs in various motion directions. The first and second row represent the images before and after feature augmentation, respectively.

#### Visualizations_OS ####

This script processes .mat files from the 'Visualizations\Showcase Datas\' directory, generating and saving visualizations of DTMs and their augmented versions using the FLM_Processing function, with output saved as .png files in specified directories.

**Input:** None (uses predefined `src_dir` for input files and `dest_dir`, `augment_dir` for output directories; requires `FLM_Processing` function and my favorite colormap).

**Output:** None (saves original and augmented DTM visualizations as .png files).

### B. Codes Explanation (None) ###

None supported codes included.

### C. Datafiles Explanation (Folder: Root, Visualizations) ###

#### 1. JoeyBG_CList.mat ####

My favorite colormap file used for generating figures in the paper.

#### 2. Three subfolders and multiple image files in "Visualizations\" ####

Eight sets of data used to generate paper visualizations, with corresponding original DTMs, and augmented DTMs.

## V. SOME THINGS TO NOTE ##

**(1) Reproducibility Issues:** All input data or images must be gray-scale maps stored in Unit8 form or Double form. At least from my pre-upload debugging, the code must not report errors as long as the folders and data are placed correctly.

**(2) Environment Issues:** The project consists of mostly MATLAB code with some python code. The recommended MATLAB version is R2024b and above. The program is executed by the CPU environment of Matlab code and GPU environment of Python code. The Python code also provides CPU running options. Of course, we recommend using GPU to speed up execution.

**(3) Algorithm Design Issues:** Adjusting the output image scale of FLM can change the amount of micro-Doppler detail information in the augmented DTM. Correspondingly, the size of the input image to the network can be adjusted. To obtain better parameter count and inference speed performance, the number of stacked feature extraction modules in the network can be reduced to two.

**(4) Right Issues: ⭐The project is limited to learning purposes only. Dataset is from open-source work of L. Du et al. Part of the FLM code is from kunzhan's open source repository. Any use or interpretation without authorized is not allowed!⭐**

Last but not least, hope that my work will bring positive contributions to the open source community in the filed of radar signal processing.
