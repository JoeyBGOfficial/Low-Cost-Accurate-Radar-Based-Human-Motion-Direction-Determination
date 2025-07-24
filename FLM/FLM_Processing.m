%% FLM_Processing Function for Image Enhancement Using FLM
% Former Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2025-7-21.
% Platform: MATLAB R2024b.
% Affiliation: Beijing Institute of Technology.
%
% Purpose:
% This function applies the Feature-Linking Model (FLM) to enhance an input image (tfmap) 
% after performing adaptive histogram equalization and thresholding. The enhanced image 
% is then resized to a specified resolution.
%
% Inputs:
%   tfmap - Input image (grayscale or RGB) to be enhanced.
%   Cutting_Threshold - Threshold value for cutting off low-intensity pixels.
%   Estimation_Resolution - Desired resolution for the output image.
%
% Outputs:
%   FLM_Enhancement - Enhanced image after FLM processing and resizing.
%
% Usage:
%   FLM_Enhancement = FLM_Processing(tfmap, Cutting_Threshold, Estimation_Resolution);
%
% Notes:
% - The function assumes tfmap is a 2D matrix (grayscale image) or 3D matrix (RGB image).
% - It requires the FLM function, which is not defined here.
% - The function also uses adapthisteq, rgb2v, and imresize from MATLAB's Image Processing Toolbox.
% - Optional quality evaluation metrics (Contrast, Spatial_frequency, Gradient) are computed but not returned.

%% Function Body
function FLM_Enhancement = FLM_Processing(tfmap,Cutting_Threshold,Estimation_Resolution)
    % Initial parameters for estimation.
    Contrast = 1; 
    Spatial_frequency = 1; 
    Gradient = 1;

    % Main process of the FLM.
    tfmap_Normalized = (tfmap - min(min(tfmap))) / (max(max(tfmap)) - min(min(tfmap))); % Normalize input tfmap.
    I = adapthisteq(tfmap_Normalized); % Adaptive histogram enhancement for the input tfmap.
    I_Normalized = (I - min(min(I))) / (max(max(I)) - min(min(I))); 
    I(I < Cutting_Threshold * max(I(:))) = 0; % Suggested threshold value: 0.62.
    V = rgb2v(I);
    V_flm = FLM(V);

    % Output the results.
    [Contrast, Spatial_frequency, Gradient] = QEvaluation(V_flm); % Optional.
    FLM_Enhancement = imresize(V_flm,[Estimation_Resolution Estimation_Resolution]);
end
