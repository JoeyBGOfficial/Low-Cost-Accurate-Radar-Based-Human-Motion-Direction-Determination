%% HMDD_Model_Construction_Transformer Function Used for Generating dlnetwork Variable
% Former Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2025.7.23.
% Platform: MATLAB R2024b.
% Affiliation: Beijing Institute of Technology.
%
% Info:
%   - This function constructs a deep learning network model for image classification
%       using a tiny Transformer-based architecture, similar to the Vision Transformer (ViT).
%       The model includes an image input layer, patch embedding, position embedding,
%       multiple Transformer encoder blocks, and a classification head.
%
% Parameters:
%   - num_classes: The number of classes for classification.
%
% Returns:
%   - HMDD_Model: A dlnetwork object representing the constructed model.
%
% Usage:
%   - Call this function with the desired number of classes to obtain the model.
%   - Ensure that the necessary custom layers (e.g., patchEmbeddingLayer,
%   positionEmbeddingLayer, etc.) are defined and accessible.
%
% Notes:
%   - The model is initialized after construction.
%   - The network plot is generated for visualization.

%% Model Body
function HMDD_Model = HMDD_Model_Construction_Transformer(num_classes)
    % Initialize the dlnetwork object.
    net = dlnetwork;
    
    % Define the initial layers: image input, validation, patch embedding, and concatenation.
    tempNet = [
        imageInputLayer([384 384 3],"Name","imageinput","Normalization","zscore")
        functionLayer(@vision.internal.cnn.errorIfInvalidInputSize,"Name",'validate_inputsize')
        patchEmbeddingLayer([16 16],192,"Name","embedding","SpatialFlattenMode","row-major")
        embeddingConcatenationLayer("Name","clsembed_concat")
        ];
    net = addLayers(net,tempNet);
    
    % Add position embedding layer.
    tempNet = positionEmbeddingLayer(192,577,"Name","posembed_input");
    net = addLayers(net,tempNet);
    
    % Add addition and dropout layers for the input to the first encoder block.
    tempNet = [
        additionLayer(2,"Name","add")
        dropoutLayer(0.1,"Name","dropout")
        ];
    net = addLayers(net,tempNet);
    
    % Define the first Transformer encoder block.
    tempNet = [
        layerNormalizationLayer("Name","encoderblock1_layernorm1","Epsilon",1e-06)
        selfAttentionLayer(3,192,"Name","encoderblock1_mha","DropoutProbability",0.1,"NumValueChannels",192,"OutputSize",192)
        dropoutLayer(0.1,"Name","encoderblock1_dropout1")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock1_add1");
    net = addLayers(net,tempNet);
    
    tempNet = [
        layerNormalizationLayer("Name","encoderblock1_layernorm2","Epsilon",1e-06)
        convolution1dLayer(1,768,"Name","encoderblock1_conv1d1")
        geluLayer("Name","encoderblock1_gelu","Approximation","tanh")
        dropoutLayer(0.1,"Name","encoderblock1_dropout2")
        convolution1dLayer(1,192,"Name","encoderblock1_conv1d2")
        dropoutLayer(0.1,"Name","encoderblock1_dropout3")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock1_add2");
    net = addLayers(net,tempNet);
    
    % Define the second Transformer encoder block.
    tempNet = [
        layerNormalizationLayer("Name","encoderblock2_layernorm1","Epsilon",1e-06)
        selfAttentionLayer(3,192,"Name","encoderblock2_mha","DropoutProbability",0.1,"NumValueChannels",192,"OutputSize",192)
        dropoutLayer(0.1,"Name","encoderblock2_dropout1")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock2_add1");
    net = addLayers(net,tempNet);
    
    tempNet = [
        layerNormalizationLayer("Name","encoderblock2_layernorm2","Epsilon",1e-06)
        convolution1dLayer(1,768,"Name","encoderblock2_conv1d1")
        geluLayer("Name","encoderblock2_gelu","Approximation","tanh")
        dropoutLayer(0.1,"Name","encoderblock2_dropout2")
        convolution1dLayer(1,192,"Name","encoderblock2_conv1d2")
        dropoutLayer(0.1,"Name","encoderblock2_dropout3")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock2_add2");
    net = addLayers(net,tempNet);
    
    % Define the third Transformer encoder block.
    tempNet = [
        layerNormalizationLayer("Name","encoderblock3_layernorm1","Epsilon",1e-06)
        selfAttentionLayer(3,192,"Name","encoderblock3_mha","DropoutProbability",0.1,"NumValueChannels",192,"OutputSize",192)
        dropoutLayer(0.1,"Name","encoderblock3_dropout1")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock3_add1");
    net = addLayers(net,tempNet);
    
    tempNet = [
        layerNormalizationLayer("Name","encoderblock3_layernorm2","Epsilon",1e-06)
        convolution1dLayer(1,768,"Name","encoderblock3_conv1d1")
        geluLayer("Name","encoderblock3_gelu","Approximation","tanh")
        dropoutLayer(0.1,"Name","encoderblock3_dropout2")
        convolution1dLayer(1,192,"Name","encoderblock3_conv1d2")
        dropoutLayer(0.1,"Name","encoderblock3_dropout3")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock3_add2");
    net = addLayers(net,tempNet);
    
    % Define the fourth Transformer encoder block.
    tempNet = [
        layerNormalizationLayer("Name","encoderblock4_layernorm1","Epsilon",1e-06)
        selfAttentionLayer(3,192,"Name","encoderblock4_mha","DropoutProbability",0.1,"NumValueChannels",192,"OutputSize",192)
        dropoutLayer(0.1,"Name","encoderblock4_dropout1")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock4_add1");
    net = addLayers(net,tempNet);
    
    tempNet = [
        layerNormalizationLayer("Name","encoderblock4_layernorm2","Epsilon",1e-06)
        convolution1dLayer(1,768,"Name","encoderblock4_conv1d1")
        geluLayer("Name","encoderblock4_gelu","Approximation","tanh")
        dropoutLayer(0.1,"Name","encoderblock4_dropout2")
        convolution1dLayer(1,192,"Name","encoderblock4_conv1d2")
        dropoutLayer(0.1,"Name","encoderblock4_dropout3")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock4_add2");
    net = addLayers(net,tempNet);
    
    % Define the fifth Transformer encoder block.
    tempNet = [
        layerNormalizationLayer("Name","encoderblock5_layernorm1","Epsilon",1e-06)
        selfAttentionLayer(3,192,"Name","encoderblock5_mha","DropoutProbability",0.1,"NumValueChannels",192,"OutputSize",192)
        dropoutLayer(0.1,"Name","encoderblock5_dropout1")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock5_add1");
    net = addLayers(net,tempNet);
    
    tempNet = [
        layerNormalizationLayer("Name","encoderblock5_layernorm2","Epsilon",1e-06)
        convolution1dLayer(1,768,"Name","encoderblock5_conv1d1")
        geluLayer("Name","encoderblock5_gelu","Approximation","tanh")
        dropoutLayer(0.1,"Name","encoderblock5_dropout2")
        convolution1dLayer(1,192,"Name","encoderblock5_conv1d2")
        dropoutLayer(0.1,"Name","encoderblock5_dropout3")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock5_add2");
    net = addLayers(net,tempNet);
    
    % Define the sixth Transformer encoder block.
    tempNet = [
        layerNormalizationLayer("Name","encoderblock6_layernorm1","Epsilon",1e-06)
        selfAttentionLayer(3,192,"Name","encoderblock6_mha","DropoutProbability",0.1,"NumValueChannels",192,"OutputSize",192)
        dropoutLayer(0.1,"Name","encoderblock6_dropout1")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock6_add1");
    net = addLayers(net,tempNet);
    
    tempNet = [
        layerNormalizationLayer("Name","encoderblock6_layernorm2","Epsilon",1e-06)
        convolution1dLayer(1,768,"Name","encoderblock6_conv1d1")
        geluLayer("Name","encoderblock6_gelu","Approximation","tanh")
        dropoutLayer(0.1,"Name","encoderblock6_dropout2")
        convolution1dLayer(1,192,"Name","encoderblock6_conv1d2")
        dropoutLayer(0.1,"Name","encoderblock6_dropout3")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock6_add2");
    net = addLayers(net,tempNet);
    
    % Define the seventh Transformer encoder block.
    tempNet = [
        layerNormalizationLayer("Name","encoderblock7_layernorm1","Epsilon",1e-06)
        selfAttentionLayer(3,192,"Name","encoderblock7_mha","DropoutProbability",0.1,"NumValueChannels",192,"OutputSize",192)
        dropoutLayer(0.1,"Name","encoderblock7_dropout1")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock7_add1");
    net = addLayers(net,tempNet);
    
    tempNet = [
        layerNormalizationLayer("Name","encoderblock7_layernorm2","Epsilon",1e-06)
        convolution1dLayer(1,768,"Name","encoderblock7_conv1d1")
        geluLayer("Name","encoderblock7_gelu","Approximation","tanh")
        dropoutLayer(0.1,"Name","encoderblock7_dropout2")
        convolution1dLayer(1,192,"Name","encoderblock7_conv1d2")
        dropoutLayer(0.1,"Name","encoderblock7_dropout3")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock7_add2");
    net = addLayers(net,tempNet);
    
    % Define the eighth Transformer encoder block.
    tempNet = [
        layerNormalizationLayer("Name","encoderblock8_layernorm1","Epsilon",1e-06)
        selfAttentionLayer(3,192,"Name","encoderblock8_mha","DropoutProbability",0.1,"NumValueChannels",192,"OutputSize",192)
        dropoutLayer(0.1,"Name","encoderblock8_dropout1")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock8_add1");
    net = addLayers(net,tempNet);
    
    tempNet = [
        layerNormalizationLayer("Name","encoderblock8_layernorm2","Epsilon",1e-06)
        convolution1dLayer(1,768,"Name","encoderblock8_conv1d1")
        geluLayer("Name","encoderblock8_gelu","Approximation","tanh")
        dropoutLayer(0.1,"Name","encoderblock8_dropout2")
        convolution1dLayer(1,192,"Name","encoderblock8_conv1d2")
        dropoutLayer(0.1,"Name","encoderblock8_dropout3")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock8_add2");
    net = addLayers(net,tempNet);
    
    % Define the ninth Transformer encoder block.
    tempNet = [
        layerNormalizationLayer("Name","encoderblock9_layernorm1","Epsilon",1e-06)
        selfAttentionLayer(3,192,"Name","encoderblock9_mha","DropoutProbability",0.1,"NumValueChannels",192,"OutputSize",192)
        dropoutLayer(0.1,"Name","encoderblock9_dropout1")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock9_add1");
    net = addLayers(net,tempNet);
    
    tempNet = [
        layerNormalizationLayer("Name","encoderblock9_layernorm2","Epsilon",1e-06)
        convolution1dLayer(1,768,"Name","encoderblock9_conv1d1")
        geluLayer("Name","encoderblock9_gelu","Approximation","tanh")
        dropoutLayer(0.1,"Name","encoderblock9_dropout2")
        convolution1dLayer(1,192,"Name","encoderblock9_conv1d2")
        dropoutLayer(0.1,"Name","encoderblock9_dropout3")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock9_add2");
    net = addLayers(net,tempNet);
    
    % Define the tenth Transformer encoder block.
    tempNet = [
        layerNormalizationLayer("Name","encoderblock10_layernorm1","Epsilon",1e-06)
        selfAttentionLayer(3,192,"Name","encoderblock10_mha","DropoutProbability",0.1,"NumValueChannels",192,"OutputSize",192)
        dropoutLayer(0.1,"Name","encoderblock10_dropout1")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock10_add1");
    net = addLayers(net,tempNet);
    
    tempNet = [
        layerNormalizationLayer("Name","encoderblock10_layernorm2","Epsilon",1e-06)
        convolution1dLayer(1,768,"Name","encoderblock10_conv1d1")
        geluLayer("Name","encoderblock10_gelu","Approximation","tanh")
        dropoutLayer(0.1,"Name","encoderblock10_dropout2")
        convolution1dLayer(1,192,"Name","encoderblock10_conv1d2")
        dropoutLayer(0.1,"Name","encoderblock10_dropout3")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock10_add2");
    net = addLayers(net,tempNet);
    
    % Define the eleventh Transformer encoder block.
    tempNet = [
        layerNormalizationLayer("Name","encoderblock11_layernorm1","Epsilon",1e-06)
        selfAttentionLayer(3,192,"Name","encoderblock11_mha","DropoutProbability",0.1,"NumValueChannels",192,"OutputSize",192)
        dropoutLayer(0.1,"Name","encoderblock11_dropout1")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock11_add1");
    net = addLayers(net,tempNet);
    
    tempNet = [
        layerNormalizationLayer("Name","encoderblock11_layernorm2","Epsilon",1e-06)
        convolution1dLayer(1,768,"Name","encoderblock11_conv1d1")
        geluLayer("Name","encoderblock11_gelu","Approximation","tanh")
        dropoutLayer(0.1,"Name","encoderblock11_dropout2")
        convolution1dLayer(1,192,"Name","encoderblock11_conv1d2")
        dropoutLayer(0.1,"Name","encoderblock11_dropout3")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock11_add2");
    net = addLayers(net,tempNet);
    
    % Define the twelfth Transformer encoder block.
    tempNet = [
        layerNormalizationLayer("Name","encoderblock12_layernorm1","Epsilon",1e-06)
        selfAttentionLayer(3,192,"Name","encoderblock12_mha","DropoutProbability",0.1,"NumValueChannels",192,"OutputSize",192)
        dropoutLayer(0.1,"Name","encoderblock12_dropout1")
        ];
    net = addLayers(net,tempNet);
    
    tempNet = additionLayer(2,"Name","encoderblock12_add1");
    net = addLayers(net,tempNet);
    
    tempNet = [
        layerNormalizationLayer("Name","encoderblock12_layernorm2","Epsilon",1e-06)
        convolution1dLayer(1,768,"Name","encoderblock12_conv1d1")
        geluLayer("Name","encoderblock12_gelu","Approximation","tanh")
        dropoutLayer(0.1,"Name","encoderblock12_dropout2")
        convolution1dLayer(1,192,"Name","encoderblock12_conv1d2")
        dropoutLayer(0.1,"Name","encoderblock12_dropout3")
        ];
    net = addLayers(net,tempNet);
    
    % Define the final layers: addition, normalization, indexing, classification head, and softmax.
    tempNet = [
        additionLayer(2,"Name","encoderblock12_add2")
        layerNormalizationLayer("Name","encoder_norm","Epsilon",1e-06)
        indexing1dLayer("first","Name","cls_index")
        fullyConnectedLayer(num_classes,"Name","head")
        softmaxLayer("Name","softmax")
        ];
    net = addLayers(net,tempNet);
    
    % Clear temporary network variable.
    clear tempNet;
    
    % Connect all the layers in the network.
    net = connectLayers(net,"clsembed_concat","posembed_input");
    net = connectLayers(net,"clsembed_concat","add/in2");
    net = connectLayers(net,"posembed_input","add/in1");
    net = connectLayers(net,"dropout","encoderblock1_layernorm1");
    net = connectLayers(net,"dropout","encoderblock1_add1/in2");
    net = connectLayers(net,"encoderblock1_dropout1","encoderblock1_add1/in1");
    net = connectLayers(net,"encoderblock1_add1","encoderblock1_layernorm2");
    net = connectLayers(net,"encoderblock1_add1","encoderblock1_add2/in2");
    net = connectLayers(net,"encoderblock1_dropout3","encoderblock1_add2/in1");
    net = connectLayers(net,"encoderblock1_add2","encoderblock2_layernorm1");
    net = connectLayers(net,"encoderblock1_add2","encoderblock2_add1/in2");
    net = connectLayers(net,"encoderblock2_dropout1","encoderblock2_add1/in1");
    net = connectLayers(net,"encoderblock2_add1","encoderblock2_layernorm2");
    net = connectLayers(net,"encoderblock2_add1","encoderblock2_add2/in2");
    net = connectLayers(net,"encoderblock2_dropout3","encoderblock2_add2/in1");
    net = connectLayers(net,"encoderblock2_add2","encoderblock3_layernorm1");
    net = connectLayers(net,"encoderblock2_add2","encoderblock3_add1/in2");
    net = connectLayers(net,"encoderblock3_dropout1","encoderblock3_add1/in1");
    net = connectLayers(net,"encoderblock3_add1","encoderblock3_layernorm2");
    net = connectLayers(net,"encoderblock3_add1","encoderblock3_add2/in2");
    net = connectLayers(net,"encoderblock3_dropout3","encoderblock3_add2/in1");
    net = connectLayers(net,"encoderblock3_add2","encoderblock4_layernorm1");
    net = connectLayers(net,"encoderblock3_add2","encoderblock4_add1/in2");
    net = connectLayers(net,"encoderblock4_dropout1","encoderblock4_add1/in1");
    net = connectLayers(net,"encoderblock4_add1","encoderblock4_layernorm2");
    net = connectLayers(net,"encoderblock4_add1","encoderblock4_add2/in2");
    net = connectLayers(net,"encoderblock4_dropout3","encoderblock4_add2/in1");
    net = connectLayers(net,"encoderblock4_add2","encoderblock5_layernorm1");
    net = connectLayers(net,"encoderblock4_add2","encoderblock5_add1/in2");
    net = connectLayers(net,"encoderblock5_dropout1","encoderblock5_add1/in1");
    net = connectLayers(net,"encoderblock5_add1","encoderblock5_layernorm2");
    net = connectLayers(net,"encoderblock5_add1","encoderblock5_add2/in2");
    net = connectLayers(net,"encoderblock5_dropout3","encoderblock5_add2/in1");
    net = connectLayers(net,"encoderblock5_add2","encoderblock6_layernorm1");
    net = connectLayers(net,"encoderblock5_add2","encoderblock6_add1/in2");
    net = connectLayers(net,"encoderblock6_dropout1","encoderblock6_add1/in1");
    net = connectLayers(net,"encoderblock6_add1","encoderblock6_layernorm2");
    net = connectLayers(net,"encoderblock6_add1","encoderblock6_add2/in2");
    net = connectLayers(net,"encoderblock6_dropout3","encoderblock6_add2/in1");
    net = connectLayers(net,"encoderblock6_add2","encoderblock7_layernorm1");
    net = connectLayers(net,"encoderblock6_add2","encoderblock7_add1/in2");
    net = connectLayers(net,"encoderblock7_dropout1","encoderblock7_add1/in1");
    net = connectLayers(net,"encoderblock7_add1","encoderblock7_layernorm2");
    net = connectLayers(net,"encoderblock7_add1","encoderblock7_add2/in2");
    net = connectLayers(net,"encoderblock7_dropout3","encoderblock7_add2/in1");
    net = connectLayers(net,"encoderblock7_add2","encoderblock8_layernorm1");
    net = connectLayers(net,"encoderblock7_add2","encoderblock8_add1/in2");
    net = connectLayers(net,"encoderblock8_dropout1","encoderblock8_add1/in1");
    net = connectLayers(net,"encoderblock8_add1","encoderblock8_layernorm2");
    net = connectLayers(net,"encoderblock8_add1","encoderblock8_add2/in2");
    net = connectLayers(net,"encoderblock8_dropout3","encoderblock8_add2/in1");
    net = connectLayers(net,"encoderblock8_add2","encoderblock9_layernorm1");
    net = connectLayers(net,"encoderblock8_add2","encoderblock9_add1/in2");
    net = connectLayers(net,"encoderblock9_dropout1","encoderblock9_add1/in1");
    net = connectLayers(net,"encoderblock9_add1","encoderblock9_layernorm2");
    net = connectLayers(net,"encoderblock9_add1","encoderblock9_add2/in2");
    net = connectLayers(net,"encoderblock9_dropout3","encoderblock9_add2/in1");
    net = connectLayers(net,"encoderblock9_add2","encoderblock10_layernorm1");
    net = connectLayers(net,"encoderblock9_add2","encoderblock10_add1/in2");
    net = connectLayers(net,"encoderblock10_dropout1","encoderblock10_add1/in1");
    net = connectLayers(net,"encoderblock10_add1","encoderblock10_layernorm2");
    net = connectLayers(net,"encoderblock10_add1","encoderblock10_add2/in2");
    net = connectLayers(net,"encoderblock10_dropout3","encoderblock10_add2/in1");
    net = connectLayers(net,"encoderblock10_add2","encoderblock11_layernorm1");
    net = connectLayers(net,"encoderblock10_add2","encoderblock11_add1/in2");
    net = connectLayers(net,"encoderblock11_dropout1","encoderblock11_add1/in1");
    net = connectLayers(net,"encoderblock11_add1","encoderblock11_layernorm2");
    net = connectLayers(net,"encoderblock11_add1","encoderblock11_add2/in2");
    net = connectLayers(net,"encoderblock11_dropout3","encoderblock11_add2/in1");
    net = connectLayers(net,"encoderblock11_add2","encoderblock12_layernorm1");
    net = connectLayers(net,"encoderblock11_add2","encoderblock12_add1/in2");
    net = connectLayers(net,"encoderblock12_dropout1","encoderblock12_add1/in1");
    net = connectLayers(net,"encoderblock12_add1","encoderblock12_layernorm2");
    net = connectLayers(net,"encoderblock12_add1","encoderblock12_add2/in2");
    net = connectLayers(net,"encoderblock12_dropout3","encoderblock12_add2/in1");
    
    % Network output.
    net = initialize(net); % Initialize the network.
    plot(net); % Plot the network for visualization.
    HMDD_Model = net; % Assign the constructed network to the output variable.
end