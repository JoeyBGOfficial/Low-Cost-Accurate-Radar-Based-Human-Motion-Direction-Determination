# Low-Cost-Accurate-Radar-Based-Human-Motion-Direction-Determination
## I. Introduction ##

### Write Sth. Upfront: ###

So this is the second project that I am publishing on a preprint platform, which is not planning to submit to a journal or conference, but is open source. 

The main purpose of this work is to provide an angle prior for radar-based human gait recognition tasks. At the same time, we hope to explore a relatively accurate and low-cost solution.

### Basic Information: ###

This repository is the open source code for my latest work: "Exploration of Low-Cost but Accurate Radar-Based Human Motion Direction Determination", submitted to arXiv.

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

#### Feature_Augmentation_OS ####

This script processes .mat files containing 'tfmap' variables from eight subfolders, applies feature augmentation using the FLM_Processing function, and saves the enhanced images as .png files with a 'jet' colormap in corresponding output subfolders.

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


### C. Datafiles Explanation (None) ###

No datafiles included.

## III. LOW-COST BUT ACCURATE HMDD METHOD ##

### A. Theory in Simple ###

Essentially, this part uses a slightly modified SBCFormer to implement augmented DTM to angle tag mapping. The proposed network is a lightweight ViT-CNN hybrid architecture with fewer model parameters and extremely high inference efficiency, while achieving high accuracy performance at the same time. The code provides two options: Both Matlab version and an improved Python version. 

![HMDD_Network](https://github.com/user-attachments/assets/dacea997-be67-4064-ac07-40b409fd7e5b)

Fig. 2. Schematic of the proposed low-cost accurate HMDD model based on a ViT-CNN hybrid architecture.

### B. Codes Explanation (Folder: ACM_Based_Micro-Doppler_Extraction) ###


#### 1. backward_gradient ####

This function computes the backward differences of a 2D matrix `f`, approximating partial derivatives along rows and columns.

**Input:** 2D matrix `f` of level set function or image.

**Output:** Two 2D matrices: `bdy` for column-wise backward differences, `bdx` for row-wise backward differences.



#### 2. Corner_Representation ####

This function detects SIFT corner points in a grayscale image, computes their centroid, and finds the pixels farthest and nearest to it.

**Input:** 2D matrix `I` of grayscale image, scalar `Corner_Threshold_Ratio`.

**Output:** Vectors `farPixel` and `nearPixel` (1x2 vector), normalized image `II_Normalized`, corner locations, and coordinates of farthest/nearest pixels.



#### 3. forward_gradient ####

This function computes the forward differences of a 2D matrix `f`, approximating partial derivatives along rows and columns.

**Input:** 2D matrix `f`.

**Output:** Two 2D matrices: `fdy` for column-wise forward differences, `fdx` for row-wise forward differences.



#### 4. EVOLUTION_4PHASE ####

This function evolves two level set functions for four-phase image segmentation using curvature and data fitting terms over multiple iterations.

**Input:** Image `I`, initial level sets `phi0`, parameters `nu`, `lambda_1`, `lambda_2`, `delta_t`, `epsilon`, and `numIter`.

**Output:** Evolved level set functions `phi` in 3D matrix.


#### 5. EVOLUTION_4PHASE_DR ####

This function extends `EVOLUTION_4PHASE` by adding a distance regularization term to maintain level set regularity.

**Input:** Same as `EVOLUTION_4PHASE`, plus `mu` for regularization weight.

**Output:** Evolved level set functions `phi` in 3D matrix.



#### 6. Delta ####

This function computes a smooth approximation of the Dirac delta function for level set methods, focused near the zero level set.

**Input:** 2D matrix `phi` represnets the level set function, scalar `epsilon` for width parameter.

**Output:** 2D matrix `Delta_h` approximating the Dirac delta function.



#### 7. get_contour ####

This function generates contour points along the perimeter of a rectangular region inset by `margin` pixels in a grid.

**Input:** Image `I` (unused), integers `nrow`, `ncol` of grid size, `margin` of inset distance.

**Output:** Arrays `xcontour` and `ycontour` represent the coordinates of contour points.



#### 8. Heaviside ####

This function computes a smooth approximation of the Heaviside step function using the arctangent function.

**Input:** 2D matrix or scalar `phi`, scalar `epsilon` for smoothness.

**Output:** Smooth Heaviside function values `H`.


#### 9. initial_sdf2circle ####

This function initializes two level set functions as circles centered at specified pixels in a grid.

**Input:** Grid size `nrow`, `ncol`, unused `ic`, `jc`, `fun_n`, radius factor `r`, center coordinates `far_pixel`, `near_pixel`.

**Output:** 3D matrix `f` containing two level set functions.


#### 10. quadrifit ####

This function computes constants `C` that best fit an image `U` across four regions defined by two level set functions, using a smooth Heaviside approximation.

**Input:** Level sets `phi` in 3D matrix, image `U`, `epsilon`, number of functions `fun_n` .

**Output:** Vector `C` regional constants, intermediate factors `mult` in 4D matrix.


#### 11. CURVATURE_CV ####

This function computes the curvature of a 2D matrix `f` using different finite difference schemes specified by `diff_scheme`.

**Input:** 2D matrix `f` of level set function or image, Integer `diff_scheme` (0, 1, or 2) selecting the difference scheme.

**Output:** 2D matrix `K` representing the approximated curvature of `f`.

### C. Datafiles Explanation (Folder: None) ###

None.


## IV. Indoor HAR Based on Point Cloud Matching ###

### A. Theory in Simple ###

The proposed method first converts the level sets corresponding to the micro-Doppler signature into contour point clouds using the MATLAB embedded contour() function. Secondly, the point cloud and the template are subjected to a similarity metric using the Mapper algorithm to obtain the final recognition results.

![Pointcloud_Matching](https://github.com/user-attachments/assets/506bf81d-1cd3-4fa8-9acd-e57a935a88e3)

Fig. 4. Schematic diagram of the proposed indoor HAR method based on point cloud topological structure similarity using Mapper algorithm.


### B. Codes Explanation (Folder: PointCloud_Matching) ###


#### 1. Mapper_Similarity ####

This function computes the topological similarity between two 2D point clouds using a simplified Mapper algorithm, measuring similarity via the Jaccard index of their Mapper graph edge sets.

**Input:** Two 2D point clouds `PC` and `PC_Class` of 2xN and 2xM matrices; Optional: integers `nx`, `ny` of grid squares, scalar `overlap_factor`.

**Output:** Scalar `similarity` represents the Jaccard similarity between edge sets.


#### 2. Select_Points_for_Columns ####

This function identifies the top points with the largest values in each column of a matrix, recording their row indices.

**Input:** 2D matrix `phi_1_Normalized` with size Estimation_Resolution x Estimation_Resolution, integer `Points_Num_Per_Column`.

**Output:** Matrix `points` in 2x(Points_Num_Per_Column*Estimation_Resolution) of row and column indices of the top points.


#### 3. contour ####

This function generates contour plots of a 2D matrix with options for specifying coordinates, levels, and line styles, supporting automatic or user-defined contour levels.

**Input:** Variable inputs: matrix `Z`, optional coordinates `X`, `Y`, levels `N` or `V`, axes handle, line specifications, and name-value pairs.

**Output:** Contour matrix `cout`, graphics handle `hand`.


#### 4. Wasserstein_Similarity ####

This function computes the 2-Wasserstein distance between two 2D point clouds using the optimal transport formulation with squared Euclidean distance costs.

**Input:** Two 2D point clouds `pointCloud1` and `pointCloud2` of 2xN and 2xM matrices.

**Output:** Scalar `similarity` represents the 2-Wasserstein distance.


### C. Datafiles Explanation (Folder: None) ###

None.

## V. Main Branch & Visualization ##

### A. Theory in Simple ###
For simulated RTM, simulated DTM, measured RTM, and measured DTM, we wrote different function scripts for the whole process of feature extraction and recognition. Main.m is used to achieve the inference prediction. The generation code of the template point cloud library and the code used to give all the visualized images in the paper are also open-sourced together.

![Simulated_Visualization](https://github.com/user-attachments/assets/38567527-71c4-42eb-8d46-34d9a8be968b)

Fig. 5. Simulated visualization results of the proposed method.

![Measured_Visualization](https://github.com/user-attachments/assets/d5a8d2b3-b9d6-4d47-9fb7-cea343d9052d)

Fig. 6. Measured visualization results of the proposed method.

### B. Codes Explanation (Folder: Root) ###


#### 1. Feature_Extraction_SimHRTM ####

This function deals with simulated RTM data, using a four-phase level set method for segmentation and creating a sparse point cloud from the extracted contour features.

**Input:** Path to the image data.

**Output:** Point cloud data of the extracted features.


#### 2. Feature_Extraction_SimHDTM ####

This function processes simulated DTM data, employing a four-phase level set method for image segmentation and generating a sparse point cloud from the extracted contour features.

**Input:** Path to the image data.

**Output:** Point cloud data representing the extracted features.


#### 3. Feature_Extraction_RWRTM ####

This function handles measured RTM data, utilizing a four-phase level set method for segmentation and producing a sparse point cloud from the contour of the extracted features.

**Input:** Path to the image data.

**Output:** Point cloud data of the extracted features.


#### 4. Feature_Extraction_RWDTM ####

This function processes measured DTM data, applying a four-phase level set method for image segmentation and generating a sparse point cloud from the contour features.

**Input:** Path to the image data.

**Output:** Point cloud data representing the extracted features.

#### 5. Main ####

This script serves as the primary interface for TWR HAR. It allows users to select data types (simulated or measured, RTM or DTM), performs feature extraction, and classifies the input based on similarity to template point clouds.

**Input:** User selection from a menu.

**Output:** Displays the predicted class name.


#### 6. Templates_Generator ####

This script generates template point clouds for simulated and measured RTM and DTM data. It processes images from predefined directories, extracts features, and saves the point clouds for use in classification.

**Input:** None, just use predefined paths.

**Output:** .mat files containing point cloud templates.


#### 7. Visualization_12_Activities ####

This script visualizes the feature extraction and recognition results for 12 activities across simulated and measured RTM and DTM data, displaying images without axes or labels.

**Input:** None, also use predefined paths.

**Output:** Visualized images saved to specified directories.


### C. Datafiles Explanation (Folder: Root, Visualizations) ###

#### 1. JoeyBG_CList.mat ####

My favorite colormap file used for generating figures in the paper.

#### 2. Class_Names.mat ####

Name strings of $12$ predefined classes of activities.

#### 3. Visualization Sub-Figures ####

High-resolution coordinate-free files for each subplot of the visualization experiments in the paper can be found in the Visualizations folder.

## VI. SOME THINGS TO NOTE ##

**(1) Reproducibility Issues:** All input images must be gray-scale maps stored in Unit8 form or Double form. In other words, it has to be a 2D matrix, preferably a square matrix. At least from my pre-upload debugging, the code must not report errors as long as the folders and data are placed correctly.

**(2) Environment Issues:** The project consists of the pure MATLAB code. The recommended MATLAB version is R2024b and above. The program is executed by the CPU environment. For the purpose of accelerating the optimization process, we recommend trying to port the code to the GPU.

**(3) Algorithm Design Issues:** If you feel that the proposed method has a lot of room for improvement in terms of feature extraction robustness, inference accuracy, and computational speed. My suggestion is: Optimize the diffusion strategy of the level set of ACM steps for radar images, and replace the Mapper algorithm as a more reasonable strategy for measuring the topological similarity of point clouds.

**(4) Right Issues: ⭐The project is limited to learning purposes only. Part of the ACM code is from Chunming Li's open source repository, contour code is borrowed from the MATLAB embedded functions, and the other content is my own original. Any use or interpretation without authorized by me is not allowed!⭐**

Last but not least, hope that my work will bring positive contributions to the open source community in the filed of radar signal processing.
