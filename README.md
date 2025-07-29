# Low-Cost-Accurate-Radar-Based-Human-Motion-Direction-Determination
## I. Introduction ##

### Write Sth. Upfront: ###

So this is the second project that I am publishing on a preprint platform, which is not planning to submit to a journal or conference, but is open source.

### Basic Information: ###

This repository is the open source code for my latest work: "Through-the-Wall Radar Human Activity Recognition WITHOUT Using Neural Networks", submitted to arXiv.

![Introduction](https://github.com/user-attachments/assets/1fdef49f-98f2-4b03-ac26-fef91f58b39c)

Fig. 1. Current works in this field take neural network-based methods as the research hotspot. This work returns to rethink the value of traditional mindsets.

**My Email:** JoeyBG@126.com;

**Abstract:** After a few years of research in the field of through-the-wall radar (TWR) human activity recognition (HAR), I found that we seem to be stuck in the mindset of training on radar image data through neural network models. The earliest related works in this field based on template matching did not require a training process, and I believe they have never died. Because these methods possess a strong physical interpretability and are closer to the basis of theoretical signal processing research. In this paper, I would like to try to return to the original path by attempting to eschew neural networks to achieve the TWR HAR task and challenge to achieve intelligent recognition as neural network models. In detail, the range-time map and Doppler-time map of TWR are first generated. Then, the initial regions of the human target foreground and noise background on the maps are determined using corner detection method, and the micro-Doppler signature is segmented using the multiphase active contour model. The micro-Doppler segmentation feature is discretized into a two-dimensional point cloud. Finally, the topological similarity between the resulting point cloud and the point clouds of the template data is calculated using Mapper algorithm to obtain the recognition results. The effectiveness of the proposed method is demonstrated by numerical simulated and measured experiments.

**Corresponding Papers:**

[1] Weicheng Gao, “Through-the-Wall Radar Human Activity Recognition WITHOUT Using Neural Networks,” arXiv (Cornell University), June. 2025, Link: https://arxiv.org/abs/2506.05169.

### Important!!! ###

**After downloading the entire repository, first unzip "Image_Templates.rar" and "Pointcloud_Templates.rar". Put your own dataset in the unzipped "Image_Templates" folder according to the subfolders' name. Next, feel free to use the code!**

## II. TWR HUMAN ECHO MODEL ##

### A. Theory in Simple ###

The proposed method first extracts the baseband signal of TWR echo by pulse compression, and then concatenates it along the slow time dimension, and the resulting image is a range-time map (RTM). The Doppler-time map (DTM) is obtained by summing all range bins of the RTM and doing the short time fourier transform (STFT) along the slow time dimension. The target image after clutter and noise suppression is obtained by doing Moving Target Indication Filtering (MTI) and Empirical Modal Decomposition (EMD) on both RTM and DTM, respectively.

![TWR_Echo_Model](https://github.com/user-attachments/assets/b2033710-afad-4ded-a4c6-5d4502d2a875)

Fig. 2. TWR human echo model and data processing.


### B. Codes Explanation (Folder: Image_Based_Data_Processing) ###


#### 1. EMD ####

This function denoises a radar RTM using EMD by discarding initial intrinsic mode functions (IMFs).

**Input:** 2D matrix `RTM`; Integer `num_discard` denotes the number of IMFs to discard.

**Output:** 2D matrix `denoised_RTM`.



#### 2. MTI ####

This function applies a simple MTI filter to a RTM by subtracting adjacent columns.

**Input:** 2D matrix `I`, here we usually use RTM for MTI processing.

**Output:** 2D matrix `filtered_I`.



#### 3. DTM_Generator ####

This function transforms a radar RTM into a DTM using the Short-Time Fourier Transform (STFT).

**Input:** 2D matrix `I`, RTM is used here; Optional parameters: `fs` for sampling frequency, `window` for window function, `noverlap` for overlap between windows, `nfft` for FFT points.

**Output:** 2D matrix `DTM`.


### C. Datafiles Explanation (Folder: Example_Datas) ###

Here includes four images as examples: one simulated RTM, one simulated DTM, one measured RTM, and one measured DTM.


## III. MICRO-DOPPLER SIGNATURE EXTRACTION BASED ON ACM ##


### A. Theory in Simple ###

The micro-Doppler foreground and noise background centers using image corner detection are first estimated. Using the two centers as starting points, the micro-Doppler signature extraction is implemented based on multiphase Chan-Vese ACM model.

![ACM_Schematic](https://github.com/user-attachments/assets/0e152377-deaf-4eed-bc82-f599f9b8c97e)

Fig. 3. Schematic diagram of the proposed ACM-based micro-Doppler signature extraction method.


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
