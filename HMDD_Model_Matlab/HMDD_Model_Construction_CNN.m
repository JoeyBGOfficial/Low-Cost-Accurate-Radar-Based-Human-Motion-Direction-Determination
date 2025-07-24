%% HMDD_Model_Construction_CNN Function Used for Generating dlnetwork Variable
% Former Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2025.7.23.
% Platform: MATLAB R2024b.
% Affiliation: Beijing Institute of Technology.
%
% Info:
%   - This function constructs a deep learning network model for image classification
%       using a convolutional neural network (CNN) architecture.
%       The model includes an image input layer, multiple convolutional blocks,
%       and a classification head.
%
% Parameters:
%   - num_classes: The number of classes for classification.
%
% Returns:
%   - HMDD_Model: A dlnetwork object representing the constructed model.
%
% Usage:
%   - Call this function with the desired number of classes to obtain the model.
%
% Notes:
%   - The model is initialized after construction.
%   - The network plot is generated for visualization.

%% Model Body
function HMDD_Model = HMDD_Model_Construction_CNN(num_classes)
    % Initialize the dlnetwork object.
    net = dlnetwork;

    % Define the initial layers: image input and first convolutional block.
    tempNet = [
        imageInputLayer([224 224 3], "Name", "input_1", "Normalization", "zscore")
        convolution2dLayer([3 3], 32, "Name", "Conv1", "Padding", "same", "Stride", [2 2])
        batchNormalizationLayer("Name", "bn_Conv1", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "Conv1_relu")
        groupedConvolution2dLayer([3 3], 1, 32, "Name", "expanded_conv_depthwise", "Padding", "same")
        batchNormalizationLayer("Name", "expanded_conv_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "expanded_conv_depthwise_relu")
        convolution2dLayer([1 1], 16, "Name", "expanded_conv_project", "Padding", "same")
        batchNormalizationLayer("Name", "expanded_conv_project_BN", "Epsilon", 0.001)
        convolution2dLayer([1 1], 96, "Name", "block_1_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_1_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_1_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 96, "Name", "block_1_depthwise", "Padding", "same", "Stride", [2 2])
        batchNormalizationLayer("Name", "block_1_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_1_depthwise_relu")
        convolution2dLayer([1 1], 24, "Name", "block_1_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_1_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Define the second convolutional block.
    tempNet = [
        convolution2dLayer([1 1], 144, "Name", "block_2_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_2_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_2_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 144, "Name", "block_2_depthwise", "Padding", "same")
        batchNormalizationLayer("Name", "block_2_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_2_depthwise_relu")
        convolution2dLayer([1 1], 24, "Name", "block_2_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_2_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Define the third convolutional block with residual connection.
    tempNet = [
        additionLayer(2, "Name", "block_2_add")
        convolution2dLayer([1 1], 144, "Name", "block_3_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_3_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_3_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 144, "Name", "block_3_depthwise", "Padding", "same", "Stride", [2 2])
        batchNormalizationLayer("Name", "block_3_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_3_depthwise_relu")
        convolution2dLayer([1 1], 32, "Name", "block_3_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_3_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Define the fourth convolutional block.
    tempNet = [
        convolution2dLayer([1 1], 192, "Name", "block_4_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_4_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_4_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 192, "Name", "block_4_depthwise", "Padding", "same")
        batchNormalizationLayer("Name", "block_4_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_4_depthwise_relu")
        convolution2dLayer([1 1], 32, "Name", "block_4_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_4_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Add residual connection for the fourth block.
    tempNet = additionLayer(2, "Name", "block_4_add");
    net = addLayers(net, tempNet);

    % Define the fifth convolutional block.
    tempNet = [
        convolution2dLayer([1 1], 192, "Name", "block_5_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_5_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_5_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 192, "Name", "block_5_depthwise", "Padding", "same")
        batchNormalizationLayer("Name", "block_5_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_5_depthwise_relu")
        convolution2dLayer([1 1], 32, "Name", "block_5_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_5_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Define the sixth convolutional block with residual connection.
    tempNet = [
        additionLayer(2, "Name", "block_5_add")
        convolution2dLayer([1 1], 192, "Name", "block_6_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_6_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_6_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 192, "Name", "block_6_depthwise", "Padding", "same", "Stride", [2 2])
        batchNormalizationLayer("Name", "block_6_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_6_depthwise_relu")
        convolution2dLayer([1 1], 64, "Name", "block_6_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_6_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Define the seventh convolutional block.
    tempNet = [
        convolution2dLayer([1 1], 384, "Name", "block_7_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_7_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_7_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 384, "Name", "block_7_depthwise", "Padding", "same")
        batchNormalizationLayer("Name", "block_7_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_7_depthwise_relu")
        convolution2dLayer([1 1], 64, "Name", "block_7_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_7_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Add residual connection for the seventh block.
    tempNet = additionLayer(2, "Name", "block_7_add");
    net = addLayers(net, tempNet);

    % Define the eighth convolutional block.
    tempNet = [
        convolution2dLayer([1 1], 384, "Name", "block_8_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_8_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_8_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 384, "Name", "block_8_depthwise", "Padding", "same")
        batchNormalizationLayer("Name", "block_8_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_8_depthwise_relu")
        convolution2dLayer([1 1], 64, "Name", "block_8_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_8_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Add residual connection for the eighth block.
    tempNet = additionLayer(2, "Name", "block_8_add");
    net = addLayers(net, tempNet);

    % Define the ninth convolutional block.
    tempNet = [
        convolution2dLayer([1 1], 384, "Name", "block_9_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_9_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_9_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 384, "Name", "block_9_depthwise", "Padding", "same")
        batchNormalizationLayer("Name", "block_9_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_9_depthwise_relu")
        convolution2dLayer([1 1], 64, "Name", "block_9_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_9_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Define the tenth convolutional block with residual connection.
    tempNet = [
        additionLayer(2, "Name", "block_9_add")
        convolution2dLayer([1 1], 384, "Name", "block_10_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_10_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_10_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 384, "Name", "block_10_depthwise", "Padding", "same")
        batchNormalizationLayer("Name", "block_10_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_10_depthwise_relu")
        convolution2dLayer([1 1], 96, "Name", "block_10_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_10_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Define the eleventh convolutional block.
    tempNet = [
        convolution2dLayer([1 1], 576, "Name", "block_11_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_11_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_11_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 576, "Name", "block_11_depthwise", "Padding", "same")
        batchNormalizationLayer("Name", "block_11_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_11_depthwise_relu")
        convolution2dLayer([1 1], 96, "Name", "block_11_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_11_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Add residual connection for the eleventh block.
    tempNet = additionLayer(2, "Name", "block_11_add");
    net = addLayers(net, tempNet);

    % Define the twelfth convolutional block.
    tempNet = [
        convolution2dLayer([1 1], 576, "Name", "block_12_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_12_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_12_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 576, "Name", "block_12_depthwise", "Padding", "same")
        batchNormalizationLayer("Name", "block_12_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_12_depthwise_relu")
        convolution2dLayer([1 1], 96, "Name", "block_12_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_12_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Define the thirteenth convolutional block with residual connection.
    tempNet = [
        additionLayer(2, "Name", "block_12_add")
        convolution2dLayer([1 1], 576, "Name", "block_13_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_13_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_13_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 576, "Name", "block_13_depthwise", "Padding", "same", "Stride", [2 2])
        batchNormalizationLayer("Name", "block_13_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_13_depthwise_relu")
        convolution2dLayer([1 1], 160, "Name", "block_13_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_13_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Define the fourteenth convolutional block.
    tempNet = [
        convolution2dLayer([1 1], 960, "Name", "block_14_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_14_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_14_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 960, "Name", "block_14_depthwise", "Padding", "same")
        batchNormalizationLayer("Name", "block_14_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_14_depthwise_relu")
        convolution2dLayer([1 1], 160, "Name", "block_14_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_14_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Add residual connection for the fourteenth block.
    tempNet = additionLayer(2, "Name", "block_14_add");
    net = addLayers(net, tempNet);

    % Define the fifteenth convolutional block.
    tempNet = [
        convolution2dLayer([1 1], 960, "Name", "block_15_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_15_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_15_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 960, "Name", "block_15_depthwise", "Padding", "same")
        batchNormalizationLayer("Name", "block_15_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_15_depthwise_relu")
        convolution2dLayer([1 1], 160, "Name", "block_15_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_15_project_BN", "Epsilon", 0.001)
        ];
    net = addLayers(net, tempNet);

    % Define the final layers: residual connection, convolution, pooling, and classification head.
    tempNet = [
        additionLayer(2, "Name", "block_15_add")
        convolution2dLayer([1 1], 960, "Name", "block_16_expand", "Padding", "same")
        batchNormalizationLayer("Name", "block_16_expand_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_16_expand_relu")
        groupedConvolution2dLayer([3 3], 1, 960, "Name", "block_16_depthwise", "Padding", "same")
        batchNormalizationLayer("Name", "block_16_depthwise_BN", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "block_16_depthwise_relu")
        convolution2dLayer([1 1], 320, "Name", "block_16_project", "Padding", "same")
        batchNormalizationLayer("Name", "block_16_project_BN", "Epsilon", 0.001)
        convolution2dLayer([1 1], 1280, "Name", "Conv_1")
        batchNormalizationLayer("Name", "Conv_1_bn", "Epsilon", 0.001)
        clippedReluLayer(6, "Name", "out_relu")
        globalAveragePooling2dLayer("Name", "global_average_pooling2d_1")
        fullyConnectedLayer(num_classes, "Name", "fc")
        softmaxLayer("Name", "softmax")
        ];
    net = addLayers(net, tempNet);

    % Clear temporary network variable.
    clear tempNet;

    % Connect all the layers in the network.
    net = connectLayers(net, "block_1_project_BN", "block_2_expand");
    net = connectLayers(net, "block_1_project_BN", "block_2_add/in2");
    net = connectLayers(net, "block_2_project_BN", "block_2_add/in1");
    net = connectLayers(net, "block_3_project_BN", "block_4_expand");
    net = connectLayers(net, "block_3_project_BN", "block_4_add/in2");
    net = connectLayers(net, "block_4_project_BN", "block_4_add/in1");
    net = connectLayers(net, "block_4_add", "block_5_expand");
    net = connectLayers(net, "block_4_add", "block_5_add/in2");
    net = connectLayers(net, "block_5_project_BN", "block_5_add/in1");
    net = connectLayers(net, "block_6_project_BN", "block_7_expand");
    net = connectLayers(net, "block_6_project_BN", "block_7_add/in2");
    net = connectLayers(net, "block_7_project_BN", "block_7_add/in1");
    net = connectLayers(net, "block_7_add", "block_8_expand");
    net = connectLayers(net, "block_7_add", "block_8_add/in2");
    net = connectLayers(net, "block_8_project_BN", "block_8_add/in1");
    net = connectLayers(net, "block_8_add", "block_9_expand");
    net = connectLayers(net, "block_8_add", "block_9_add/in2");
    net = connectLayers(net, "block_9_project_BN", "block_9_add/in1");
    net = connectLayers(net, "block_10_project_BN", "block_11_expand");
    net = connectLayers(net, "block_10_project_BN", "block_11_add/in2");
    net = connectLayers(net, "block_11_project_BN", "block_11_add/in1");
    net = connectLayers(net, "block_11_add", "block_12_expand");
    net = connectLayers(net, "block_11_add", "block_12_add/in2");
    net = connectLayers(net, "block_12_project_BN", "block_12_add/in1");
    net = connectLayers(net, "block_13_project_BN", "block_14_expand");
    net = connectLayers(net, "block_13_project_BN", "block_14_add/in2");
    net = connectLayers(net, "block_14_project_BN", "block_14_add/in1");
    net = connectLayers(net, "block_14_add", "block_15_expand");
    net = connectLayers(net, "block_14_add", "block_15_add/in2");
    net = connectLayers(net, "block_15_project_BN", "block_15_add/in1");

    % Network output.
    net = initialize(net); % Initialize the network.
    plot(net); % Plot the network for visualization.
    HMDD_Model = net; % Assign the constructed network to the output variable.
end