%% Visualization Script for Dataset
% Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2025.7.27.
% Platform: MATLAB R2024b.
% Affiliation: Beijing Institute of Technology.
%
% This script processes .mat files from the 'Showcase Datas' directory, generating visualizations
% for Doppler-Time Maps (DTMs) and their augmented versions using the FLM_Processing function.
% Each .mat file corresponds to a specific angle (000, 030, 045, 060, 090, 300, 315, 330)
% and is expected to contain a single variable representing the DTM data.
% The script saves the visualizations as .png files in specified output directories using a custom colormap.
%
% Parameters:
%   - Sliding_Frame_Window: Size of the sliding window for processing, set to 4.
%   - Max_Frequency_of_DTM: Maximum frequency for DTM visualization, set to 200 Hz.
%   - Cutting_Threshold: Threshold for input image detection, suggested value: 0.62.
%   - Estimation_Resolution: Image resize scale for feature dataset generation, suggested value: 64.
%
% Usage:
%   - Set src_dir to the path containing the input .mat files.
%   - Set dest_dir and augment_dir to the desired output directories for original and augmented DTM visualizations.
%   - Ensure the FLM_Processing function and slanCL colormap are defined and accessible.
%   - Run the script to generate and save the DTM visualizations.
%
% Notes:
%   - The script creates output directories if they do not exist.
%   - Visualizations are saved with a resolution of 800 DPI.
%   - The script skips files with multiple variables or non-existent files, issuing warnings accordingly.

%% Initialization of Matlab Script
clear all;
close all;
clc;
CList=slanCL(1668); % My favorite colormap preset.
Class_Names = load("Class_Names.mat").Class_Names;

%% Basic Definitions
% Define necessary parameters.
Sliding_Frame_Window = 2; % The slow time window for each DTM is 2 seconds.
Max_Frequency_of_DTM = 3080; % For 77 GHz center frequency, 6 m/s target leads to 3080 Hz Doppler frequency.
Cutting_Threshold = 0.62; % Threshold detection of input image, suggested value: 0.62
Estimation_Resolution = 64; % Image resize scale for feature dataset generation, suggested value: 224/64.

% Define the list of angles.
angles = {'000', '030', '045', '060', '090', '300', '315', '330'};

% Define the source and destination directories.
src_dir = 'Visualizations\Showcase Datas\';
dest_dir = 'Visualizations\Original DTMs\';
augment_dir = 'Visualizations\Augmented DTMs\';

% Check if the destination directory exists; if not, create it.
if ~exist(dest_dir, 'dir')
    mkdir(dest_dir);
end

%% Main Loop of Visualization
% Loop through each angle.
for i = 1:length(angles)
    angle = angles{i};
    filename = fullfile(src_dir, ['001-NM-01-' angle '.MAT']);
    
    % Check if the file exists before loading.
    if exist(filename, 'file')
        % Load the .mat file into a struct.
        S = load(filename);
        var_name = fieldnames(S);
        
        % Ensure there is exactly one variable in the file.
        if length(var_name) == 1
            % Datas readin.
            data = S.(var_name{1});
            [Horiz Vert] = size(data);
            Resize_Shape = min(Horiz,Vert);

            % ----------------------------- Generate DTM for visualization. -----------------------------
            DTM = imresize(data,[Resize_Shape,Resize_Shape]);
            
            % Create a new figure and display the original DTM datas.
            figure(1);
            imagesc(flip(DTM));
            axis tight;
            colormap(flipud(CList));
            % colorbar;
            % xlabel('Time (s)');
            % ylabel('Doppler (Hz)');
            set(gca,'XTick',0:Resize_Shape/3:Resize_Shape);
            set(gca,'XTicklabel',{'','','','',''});
            set(gca,'YTick',0:Resize_Shape/3:Resize_Shape);
            set(gca,'YTicklabel',{'','','','',''});
            set(gca,'FontName','TsangerYuMo W03');
            set(gca,'FontSize',16);
            set(gca,'ydir','normal');
            % title('DTM First Frame','FontName','TsangerYuMo W03','FontWeight','bold','FontSize',20);
            set(gcf,'Position',[400, 200, 500, 600]);
            
            % Construct the output file name and save the figure.
            output_file = fullfile(dest_dir, ['Original_DTM_' angle '.png']);
            exportgraphics(gcf,output_file,"Resolution",800);

            % ----------------------------- Perform FLM feature augmentation. -----------------------------
            DTM_Augmented = FLM_Processing(data,Cutting_Threshold,Estimation_Resolution); % Main step of feature augmentation.

            % Create a new figure and display the original DTM datas.
            figure(2);
            imagesc(flip(DTM_Augmented));
            axis tight;
            colormap(flipud(CList));
            % colorbar;
            % xlabel('Time (s)');
            % ylabel('Doppler (Hz)');
            set(gca,'XTick',0:Estimation_Resolution/3:Estimation_Resolution);
            set(gca,'XTicklabel',{'','','','',''});
            set(gca,'YTick',0:Estimation_Resolution/3:Estimation_Resolution);
            set(gca,'YTicklabel',{'','','','',''});
            set(gca,'FontName','TsangerYuMo W03');
            set(gca,'FontSize',16);
            set(gca,'ydir','normal');
            % title('DTM First Frame','FontName','TsangerYuMo W03','FontWeight','bold','FontSize',20);
            set(gcf,'Position',[400, 200, 500, 600]);

            % Construct the output file name and save the figure.
            output_file = fullfile(augment_dir, ['Augmented_DTM_' angle '.png']);
            exportgraphics(gcf,output_file,"Resolution",800);
        else
            warning(['File ' filename ' contains multiple variables. Skipping.']);
        end
    else
        warning(['File ' filename ' does not exist.']);
    end
end