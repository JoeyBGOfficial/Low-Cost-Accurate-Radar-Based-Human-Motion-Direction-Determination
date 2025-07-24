%% v2rgb Function for Combining V Channel with Original Image
% Former Author: Kun Zhan, Jicai Teng, Jinhui Shi.
% Improved By: JoeyBG.
% Date: 2025-7-21.
% Platform: MATLAB R2024b.
% Affiliation: Beijing Institute of Technology.
%
% Purpose:
% This function combines the processed V channel with the original image's H and S channels
% to generate an enhanced RGB image. If the original image is grayscale, it directly returns
% the processed V channel.
%
% Inputs:
%   I - Original image, which can be either RGB or grayscale.
%   V - Processed V channel image.
%
% Outputs:
%   Io - Enhanced RGB image, or the processed V channel if the original image is grayscale.
%
% Usage:
%   Io = v2rgb(I, V);
%
% Notes:
% - The function assumes the input images I and V are of type uint8 or can be converted to uint8.
% - For RGB images, it converts the image to HSV, replaces the V channel with the processed V,
%   and converts back to RGB, scaling to [0, 255].
% - If the original image is grayscale, it directly returns the processed V channel.

%% Function Body
function Io = v2rgb(I, V)
    % Get the size of the image to determine the number of channels.
    [~, ~, hei] = size(I);  % hei represents the number of channels (1 for grayscale, 3 for RGB).
    
    % Check if the image is grayscale
    if hei == 1
        Io = V;
    else
        Io = rgb2hsv(I);
        Io(:, :, 3) = im2double(V);
        Io = uint8(hsv2rgb(Io) .* 255);
    end
end