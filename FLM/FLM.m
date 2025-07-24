%% FLM Function for Image Enhancement
% Former Author: Kun Zhan, Jicai Teng, Jinhui Shi.
% Improved By: JoeyBG.
% Date: 2025-7-21.
% Platform: MATLAB R2024b.
% Affiliation: Beijing Institute of Technology.
%
% Purpose:
% This function implements the Feature-Linking Model (FLM) for enhancing grayscale images,
% based on the method described in the paper "Feature-linking model for image enhancement"
% by K. Zhan, J. Shi, J. Teng, Q. Li, and M. Wang, Neural Computation, vol. 28, no. 6,
% pp. 1072-1100, 2016.
%
% Inputs:
%   I - Input grayscale image.
%
% Outputs:
%   Rep1gs - Enhanced image after applying FLM and gray stretching.
%
% Usage:
%   Rep1gs = FLM(I);
%
% Notes:
% - The function assumes I is a grayscale image.
% - It requires the GrayStretch function, which is not defined here.
% - The algorithm iteratively processes the image until all pixels meet a condition.

%% Function Body
function Rep1gs = FLM(I)
    % Define helper functions and params.
    funInverse = @(x) max(max(x)) + 1 - x; % Inverse image pixel value.        
    funNormalize = @(x) (x - min(min(x))) / (max(max(x)) - min(min(x)) + eps);  % Normalize image to [0,1].
    [r, c] = size(I);                            
    rc = r * c;                                   
    S = funNormalize(double(I)) + 1/255;         
    W = [0.7 1 0.7; 1  0  1; 0.7 1 0.7];         

    % Initialize variables for the algorithm.
    Y = zeros(r, c);                             
    U = zeros(r, c);                             
    Time = zeros(r, c);                           
    sumY = 0;                                    
    n = 0;                                       

    % Compute initial threshold using Laplacian filter.
    Lap = fspecial('laplacian', 0.2); % Create Laplacian filter.
    Theta = 1 + imfilter(S, Lap, 'symmetric');    

    % Compute feature map f.
    f = 0.75 * exp(-(S).^2 / 0.16) + 0.05;       
    G = fspecial('gaussian', 7, 1); % Gaussian filter for smoothing.
    f = imfilter(f, G, 'symmetric');              

    % Set parameters for the algorithm.
    h = 2e10;                                     
    d = 2;                                        
    g = 0.9811;                                  
    alpha = 0.01;                                 
    beta = 0.03;                                  

    % Iterative process of the FLM algorithm.
    while sumY < rc                               
        n = n + 1;                                
        K = conv2(Y, W, 'same');                  
        Wave = alpha * K + beta .* S .* (K - d);  
        U = f .* U + S + Wave;                    
        Theta = g * Theta + h .* Y;               
        Y = double(U > Theta);                    
        Time = Time + n .* Y;                     
        sumY = sumY + sum(Y(:));               
    end

    % Post-process to generate enhanced image.
    Rep = funInverse(Time);                       
    Rep = funNormalize(Rep);                      
    Rep = uint8(Rep * 255);                       
    Rep1gs = GrayStretch(Rep, 0.98);         
end