%% GrayStretch Function for Image Enhancement
% Former Author: Kun Zhan, Jicai Teng, Jinhui Shi.
% Improved By: JoeyBG.
% Date: 2025-7-21.
% Platform: MATLAB R2024b.
% Affiliation: Beijing Institute of Technology.
%
% Purpose:
% This function performs gray-level stretching on an input grayscale image to enhance its contrast.
% It adjusts pixel intensities based on histogram bounds determined by a specified percentage,
% mapping them to the full 0-255 range for improved visibility.
%
% Inputs:
%   I   - Input grayscale image (uint8)
%   Per - Percentage of histogram to stretch (e.g., 0.98 for 98%)
%
% Outputs:
%   GS  - Stretched grayscale image (uint8)
%
% Usage:
%   GS = GrayStretch(I, Per);
%
% References:
%   [1] K. Zhan, J. Shi, Q. Li, J. Teng, M. Wang, "Image segmentation using fast linking SCM," 
%       in Proc. of IJCNN, vol. 25, pp. 2093-2100, 2015.
%   [2] K. Zhan, J. Shi, J. Teng, Q. Li and M. Wang, "Feature-linking model for image enhancement," 
%       Neural Computation, vol. 28, no. 6, pp. 1072-1100, 2016.
%
% Notes:
% - Assumes input image I is of type uint8.
% - Relies on helper functions FindingMm and BoundFinding for bound computation.

%% Function Body
function GS = GrayStretch(I, Per)
    % Determine min and max bounds for stretching.
    [m, M] = FindingMm(I, Per);
    
    % Perform linear stretching to full 0-255 range.
    GS = uint8((double(I) - m) ./ (M - m) * 255);
end

%% FindingMm Supporting Function
function [minI, MaxI] = FindingMm(I, Per)
    % Calculate image histogram and total pixel count.
    h = imhist(I);
    All = sum(h);
    ph = h ./ All;
    mth_ceiling = BoundFinding(ph, Per);    
    
    % Reverse histogram to find lower bound for maximum intensity.
    Mph = fliplr(ph')';
    Mth_floor = BoundFinding(Mph, Per);
    Mth_floor = 256 - Mth_floor + 1; % Adjust to original scale.
    
    % Define constraint function for valid bounds.
    ConstraintJudge = @(x, y) sum(h(x:y)) / All >= Per;
    
    % Initialize difference matrix with infinity.
    Difference = zeros(256, 256) + inf;
    
    % Compute differences for valid min/max pairs.
    for m = mth_ceiling:-1:1
        for M = Mth_floor:256
            if (h(m) > 0) && (h(M) > 0)
                if ConstraintJudge(m, M)
                    Difference(m, M) = M - m;
                end
            end
        end
    end
    
    % Find smallest difference satisfying constraint.
    minD = min(Difference(:));
    [m, M] = find(Difference == minD);
    
    % Assign final min and max intensity values.
    minI = m(1) - 1;
    MaxI = M(1) - 1;
end

%% BoundFinding Supporting Function
function m_ceiling = BoundFinding(ph, Per)
    % Compute cumulative probability distribution.
    cumP = cumsum(ph);
    
    % Initialize index and residual probability.
    n = 1;
    residualP = 1 - Per;
    
    % Find smallest index where cumulative probability exceeds residual.
    while cumP(n) < residualP
        n = n + 1;
    end
    
    % Set ceiling for minimum bound.
    m_ceiling = n;
end