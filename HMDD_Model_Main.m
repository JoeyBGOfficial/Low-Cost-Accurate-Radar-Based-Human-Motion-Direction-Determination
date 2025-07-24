%% HMDD Model Training and Evaluation
% Former Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2025.7.23.
% Platform: MATLAB R2024b.
% Affiliation: Beijing Institute of Technology.
%
% Info: 
%   - This script performs training and evaluation of the HMDD model using the
%       'Feature_Enhanced_Dataset_OS' dataset. It includes data loading, preprocessing,
%       model construction, training, and evaluation with visualization of results.
%
% Functionality:
%   - Loads the 'Feature_Enhanced_Dataset_OS' dataset.
%   - Preprocesses images by resizing to 384x384/224×224 pixels.
%   - Applies Z-score normalization via the model's imageInputLayer.
%   - Monitors training progress with real-time loss and accuracy plots.
%   - Trains and evaluates the model using the HMDD_Model_Construction function.
%
% Usage:
%   - Set the datasetPath to the location of the 'Feature_Enhanced_Dataset_OS' dataset.
%   - Ensure the HMDD_Model_Construction functions are available in the MATLAB path.
%   - Run the script to train the model and evaluate its performance.
%
% Notes:
%   - The script checks for the existence of the dataset folder and required functions.
%   - Training progress is visualized in a pop-up window.
%   - Evaluation includes accuracy calculation and confusion matrix visualization.
%
% P.S. After multiple attempts, the transformer-based network's performance is still poor. 
%       So we try other architectures and build an improved Python version. 
%       However, we still decide to make the network's code publicly available.
%       Maybe it is useful for some other datasets or tasks. Who knows.

%% Initialization of Matlab Script
clear all;
close all;
clc;
disp("---------- Author: © JoeyBG © ----------"); % Display author of the script.

%% Choose to use CNN or Transformer-Based Network Structure
Choose_Structure = 'CNN'; % Choices: Only 'CNN' and 'Transformer' are supported.

%% Basic Definitions and Preparations
% Define the dataset root directory.
datasetPath = 'Open_Source_Dataset\Feature_Enhanced_Dataset_OS';
% disp(['Dataset Path: ', datasetPath]); % Check the path of the dataset.
% Verify dataset folder existence.
if ~isfolder(datasetPath)
    error('Error: Dataset folder "%s" does not exist. Please check the path.', datasetPath);
end

% Define input image size for the model.
if strcmp(Choose_Structure,'CNN')
    imageSize = [224, 224, 3]; % Hyperparameter for CNN-based structure.
elseif strcmp(Choose_Structure,'Transformer')
    imageSize = [384, 384, 3]; % Hyperparameter for transformer-based structure.
else
    disp('Structure not supported! Use default CNN-based structure instead.');
    imageSize = [224, 224, 3];
end

%% Loading and Preparing Dataset
disp('Loading dataset...');

% Dataset loading.
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Extract class names and count.
classNames = categories(imds.Labels);
numClasses = numel(classNames);
fprintf('Dataset loaded successfully.\n');

% Split dataset into training and validation sets.
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
fprintf('Dataset split completed:\n');
fprintf(' - Training set samples: %d;\n', numel(imdsTrain.Files));
fprintf(' - Validation set samples: %d.\n', numel(imdsValidation.Files));

% Set up data augmentation for image resizing.
disp('----------------------------------------');
disp(['Resize images to ', num2str(imageSize(1)), 'x', num2str(imageSize(2)), '...']);
augimdsTrain = augmentedImageDatastore(imageSize, imdsTrain);
augimdsValidation = augmentedImageDatastore(imageSize, imdsValidation);
disp('Image preprocessing configured.');

%% Constructing Network Model
disp('----------------------------------------');
disp('Building network model...');

% Check for HMDD_Model_Construction function.
if ~exist('HMDD_Model_Construction_Transformer', 'file')
    error('Error: "HMDD_Model_Construction_Transformer.m" function file not found. Ensure it is in the MATLAB path.');
end

% Define input image size for the model.
if strcmp(Choose_Structure,'CNN')
    dlnet = HMDD_Model_Construction_CNN(numClasses); % CNN-Based version of HMDD model.
elseif strcmp(Choose_Structure,'Transformer')
    dlnet = HMDD_Model_Construction_Transformer(numClasses); % Transformer-Based version of HMDD model.
else
    disp('Structure not supported! Use default CNN-based structure instead.');
    dlnet = HMDD_Model_Construction_CNN(numClasses);
end
disp('Network model construction completed.');

%% Defining Training Options
disp('----------------------------------------');
disp('Configuring training parameters...');

% Define training hyperparameters.
maxEpochs = 80; % Total training epochs.
miniBatchSize = 16; % Can be adjustable according to GPU memory.
initialLearningRate = 0.0005; % Initial learning rate for training, suggested value: 1e-4 ~ 5e-4.
% validationFrequency = floor(numel(augimdsTrain.Files) / miniBatchSize); % Validate per epoch.
validationFrequency = 40; % Validate per 40 batches.

% Set training options for the Adam optimizer.
options = trainingOptions('adam', ...
    'InitialLearnRate', initialLearningRate, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', validationFrequency, ...
    'Plots', 'training-progress', ... % Display real-time loss plots.
    'Verbose', true, ...
    'OutputNetwork', 'best-validation-loss');
disp('Training parameters configured.');

%% Training the Network
disp('----------------------------------------');
fprintf('Starting network training...\n\n');

% Main stage of network training.
[trainedNet, traininfo] = trainnet(augimdsTrain, dlnet, 'crossentropy', options);
disp('Training completed!');

%% Evaluating Model Performance
disp('----------------------------------------');
disp('Evaluating model performance on validation set...');

% Create minibatchqueue, which extracts the first column of image datas in the datastore.
% Automatically convert to 'SSCB' (Spatial, Spatial, Channel, Batch) format.
mbqValidation = minibatchqueue(augimdsValidation, 1, ...
    'MiniBatchSize', miniBatchSize, ...
    'MiniBatchFormat', 'SSCB');

% Initializing predictions.
YScores = []; 
reset(mbqValidation);

% Run for every minibatch.
while hasdata(mbqValidation)
    dlX = next(mbqValidation);
    
    % Prediction and recording.
    scoresBatch = predict(trainedNet, dlX);
    YScores = [YScores, extractdata(scoresBatch)];
end

% Find the maximum score index for each image.
[~, YPredIdx] = max(YScores, [], 1);

% Find the prediction label and convert into categorical array.
YPred = categorical(classNames(YPredIdx)); 

% Get true labels.
YValidation = imdsValidation.Labels; 

% Accuracy calculation.
accuracy = mean(YPred == YValidation);
fprintf('Final validation accuracy: %.2f%%\n', accuracy * 100);

% Plot confusion matrix for validation set.
figure('Name', 'Validation Confusion Matrix', 'NumberTitle', 'off');
cm = confusionchart(YValidation, YPred);
cm.Title = sprintf('Validation Confusion Matrix (Accuracy: %.2f%%)', accuracy * 100);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';