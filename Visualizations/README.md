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
