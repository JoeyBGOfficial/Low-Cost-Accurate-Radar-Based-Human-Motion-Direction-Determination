%% Open-Source Dataset Reconstruction for the Task of Human Motion Determination
% Former Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2025.7.21.
% Platform: MATLAB R2024b.
% Affiliation: Beijing Institute of Technology.
%
% Purpose:
% This script traverses 121 subfolders and their secondary subfolders under a specified path,
% identifies .mat files, and copies them into one of eight target folders (000, 030, 045, 060, 
% 090, 300, 315, 330) based on the three-digit direction code at the end of each filename.
%
% Usage:
% - Set the main_path and target_path variables to the desired directories.
% - Run the script to sort and copy .mat files.
%
% Notes:
% - Files with invalid direction codes are skipped with a warning.
% - Target folders are created automatically if they do not exist.

%% Initialization of Matlab Script
clear all;
close all;
clc;
disp("---------- Author: © JoeyBG © ----------"); % Display author of the script.

%% Path and Parameter Setup
main_path = 'Open_Source_Dataset\Original_MAT_Dataset_OS'; % Source directory containing subfolders 001-121.
target_path = 'Open_Source_Dataset\Direction_Dataset_OS'; % Destination directory for sorted files.
valid_directions = {'000', '030', '045', '060', '090', '300', '315', '330'}; % List of valid direction codes.

%% Subfolder Traversal
for i = 1:121 % Loop through subfolders 001 to 121.
    subfolder = sprintf('%03d', i); % Format subfolder name (e.g., 001, 002, ..., 121).
    full_subfolder_path = fullfile(main_path, subfolder); % Construct full path to subfolder.
    
    % Retrieve secondary subfolders (Including: NM-01~06, CT-01\02, BG-01\02).
    second_level_folders = dir(full_subfolder_path);
    second_level_folders = second_level_folders([second_level_folders.isdir]); % Filter for directories only.
    second_level_folders = second_level_folders(~ismember({second_level_folders.name}, {'.', '..'})); % Exclude '.' and '..'.
    
    for j = 1:length(second_level_folders) % Loop through each secondary subfolder.
        second_folder = second_level_folders(j).name; % Get secondary subfolder name.
        full_second_folder_path = fullfile(full_subfolder_path, second_folder); % Construct full path.
        
        mat_files = dir(fullfile(full_second_folder_path, '*.MAT')); % List all .mat files in the secondary subfolder.
        
        for k = 1:length(mat_files) % Loop through each .mat file.
            mat_file = mat_files(k).name; % Get filename.
            
            % Extract direction code (three digits after the last hyphen).
            idx = strfind(mat_file, '-');
            last_hyphen = idx(end);
            dot_idx = strfind(mat_file, '.');
            dot_idx = dot_idx(end);
            direction = mat_file(last_hyphen+1:dot_idx-1);
            
            % Validate direction code.
            if ~ismember(direction, valid_directions)
                warning('Skipping file %s: Invalid direction code %s', mat_file, direction);
                continue; % Skip to next file if direction is invalid.
            end
            
            % Construct target folder path.
            target_folder = fullfile(target_path, direction);
            
            % Create target folder if it doesn’t exist.
            if ~exist(target_folder, 'dir')
                mkdir(target_folder);
            end
            
            % Copy file to target folder.
            source_file = fullfile(full_second_folder_path, mat_file);
            copyfile(source_file, target_folder);
        end
    end
end