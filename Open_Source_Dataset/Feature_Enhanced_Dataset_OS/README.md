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
