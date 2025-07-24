%% FLM Function for Image Enhancement
% Former Author: Kun Zhan, Jicai Teng, Jinhui Shi.
% Improved By: JoeyBG.
% Date: 2025-7-21.
% Platform: MATLAB R2024b.
% Affiliation: Beijing Institute of Technology.
%
% Purpose:
% This function converts an RGB image to its value (V) component in the HSV color space.
% If the input image is already grayscale (single channel), it returns the image unchanged.
%
% Inputs:
%   I - Input image, which can be either RGB or grayscale.
%
% Outputs:
%   Iv - The V component of the HSV color space for RGB images, or the original image if grayscale.
%
% Usage:
%   Iv = rgb2v(I);
%
% Notes:
% - The function assumes the input image I is of type uint8 or can be converted to uint8.
% - For RGB images, it uses MATLAB's rgb2hsv function to convert to HSV and extracts the V channel,
%   scaling it to the range [0, 255].
% - The output Iv is always of type uint8.

%% Function Body
function Iv = rgb2v(I)
    % Ensure the input image is of type uint8.
    I = uint8(I);
    
    % Get the size of the image to determine the number of channels.
    [~, ~, hei] = size(I);  % hei represents the number of channels (1 for grayscale, 3 for RGB).
    
    % Check if the image is grayscale
    if hei == 1
        Iv = I;
    else
        hsv = rgb2hsv(I);
        V = hsv(:, :, 3);
        Iv = uint8(V .* 255);
    end
end