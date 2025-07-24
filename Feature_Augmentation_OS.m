%% Feature Augmentation for Open Source Dataset
% Former Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2025.7.21.
% Platform: MATLAB R2024b.
% Affiliation: Beijing Institute of Technology.
%
% This script processes .mat files from subfolders '000', '030', '045', '060', '090', '300', '315', '330'
% under the specified root_dir. Each .mat file is expected to contain a variable 'tfmap'.
% The script applies feature augmentation using the FLM_Processing function and saves the resulting
% images as .png files with 'jet' colormap in corresponding subfolders under output_root.
%
% Parameters:
% - Cutting_Threshold: Threshold for input image detection, suggested value: 0.62.
% - Estimation_Resolution: Image resize scale for feature dataset generation, suggested value: 64.
%
% Usage:
% - Set root_dir to the path containing the input dataset subfolders.
% - Set output_root to the desired output directory.
% - Ensure the FLM_Processing function is defined and accessible.
% - Run the script to generate and save the augmented feature images.
%
% Notes:
% - The script creates output subfolders if they do not exist.
% - The output images are named the same as the input .mat files but with .png extension.

%% Initialization of Matlab Script
clear all;
close all;
clc;
disp("---------- Author: © JoeyBG © ----------"); % Display author of the script.

%% Basic Definitions
% Parameter definition for feature augmentation.
Cutting_Threshold = 0.62; % Threshold detection of input image, suggested value: 0.62
Estimation_Resolution = 64; % Image resize scale for feature dataset generation, suggested value: 224/64.

% Path definition for datasets.
root_dir = 'Open_Source_Dataset\Direction_Dataset_OS';
output_root = 'Open_Source_Dataset\Feature_Enhanced_Dataset_OS';
subfolders = {'000', '030', '045', '060', '090', '300', '315', '330'};

%% Main Process of Feature Augmentation
for i = 1:length(subfolders)
    % Readin subfolders for input dataset.
    subfolder = subfolders{i};
    input_subfolder = fullfile(root_dir, subfolder);
    output_subfolder = fullfile(output_root, subfolder);
    if ~exist(output_subfolder, 'dir')
        mkdir(output_subfolder); % Make sure the output path exists.
    end
    mat_files = dir(fullfile(input_subfolder, '*.mat'));

    % Feature augmentation in sequence.
    for j = 1:length(mat_files)
        % Load the tfmap datas.
        mat_file = mat_files(j).name;
        data = load(fullfile(input_subfolder, mat_file));
        tfmap = data.tfmap;

        % Use FLM_Processing function for feature augmentation.
        FLM_Enhancement = FLM_Processing(tfmap,Cutting_Threshold,Estimation_Resolution);

        % Rescale the resulting gray-scale map.
        scaled = mat2gray(FLM_Enhancement);
        scaled = scaled * 63 + 1; % Scale to [1,64].
        scaled = round(scaled);

        % Store the map with RGB three-channel "jet" colormap.
        cmap = jet(64);
        rgb_image = ind2rgb(scaled, cmap);
        [~, name, ~] = fileparts(mat_file);
        image_name = [name '.png'];
        imwrite(rgb_image, fullfile(output_subfolder, image_name));
    end
end