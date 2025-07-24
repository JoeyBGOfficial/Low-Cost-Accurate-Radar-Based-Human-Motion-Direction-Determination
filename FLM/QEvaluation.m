%% QEvaluation Function for Image Quality Assessment
% Former Author: Kun Zhan, Jicai Teng, Jinhui Shi.
% Improved By: JoeyBG.
% Date: 2025-7-21.
% Platform: MATLAB R2024b.
% Affiliation: Beijing Institute of Technology.
%
% Purpose:
% This function assesses the quality of an enhanced grayscale image by computing
% three metrics: Local Contrast (lc), Spatial Frequency (SF), and Mean Gradient (MG).
%
% Inputs:
%   I - Input enhanced grayscale image (uint8 or double).
%
% Outputs:
%   lc - Local contrast, measuring local intensity variations.
%   SF - Spatial frequency, quantifying overall texture or activity.
%   MG - Mean gradient, representing average edge strength.
%
% Usage:
%   [lc, SF, MG] = QEvaluation(I);
%
% References:
%   [1] K. Zhan, J. Shi, J. Teng, Q. Li and M. Wang, "Feature-linking model for image enhancement," 
%       Neural Computation, vol. 28, no. 6, pp. 1072-1100, 2016.
%   [2] De Vries F P P. "Automatic, adaptive, brightness independent contrast enhancement,"
%       Signal Processing, 21(2): 169-182, 1990.
%   [3] Eskicioglu A M, Fisher P S. "Image quality measures and their performance"
%       IEEE Transactions on Communications, 43(12): 2959-2965, 1995.
%   [4] Bai X, Zhang Y. "Enhancement of microscopy mineral images through constructing 
%       alternating operators using opening and closing based toggle operator"
%       Journal of Optics, 16(12): 125407, 2014.
%
% Notes:
% - Assumes I is a 2D grayscale image; convert color images to grayscale prior to use.
% - Converts input to double precision for accurate computation.
% - Uses symmetric padding to handle edge effects in filtering.

%% Function Body
function [lc, SF, MG] = QEvaluation(I)
    % Convert image to double precision for calculations.
    I = double(I);
    [M, N] = size(I);
    Tp = M * N;

    % Local contrast (LC) estimation.
    lm = imfilter(I, ones(3)/9, 'symmetric');
    lv = imfilter(I.^2, ones(3)/9, 'symmetric') - lm.^2;
    lc = lv ./ (lm + eps);
    lc = mean2(abs(lc));

    % Spatial frequency (SF) estimation.
    RF = diff(I, 1, 2);
    Row_Freq = sqrt(sum(sum(RF.^2)) / Tp);
    CF = diff(I, 1, 1);
    Column_Freq = sqrt(sum(sum(CF.^2)) / Tp);
    SF = sqrt(Row_Freq^2 + Column_Freq^2);

    % Mean gradient (MG) estimation.
    [Fx, Fy] = gradient(I);
    G = sqrt(Fx.^2 + Fy.^2);
    MG = mean2(G);
end