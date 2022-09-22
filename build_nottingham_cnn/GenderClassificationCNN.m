%% read in the files

%Training dataset
trainingDatasetPath = fullfile(cd,'dataset');
trainingData = imageDatastore(trainingDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');


%% show a image as test

figure;

    imshow(trainingData.Files{2});
    
 %% check the number of files in each category
 labelCount = countEachLabel(trainingData)
%% determine the size of each image
img = readimage(trainingData,1);
size(img)



%% Define network 
layers = [
    imageInputLayer([30 30 1])

    convolution2dLayer(3,16,'Padding',2)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(3,32,'Padding',2)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(3,64,'Padding',2)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
      
      %%
options = trainingOptions('sgdm',...
    'InitialLearnRate',0.01,...
    'Plots','training-progress');
%%
net = trainNetwork(trainingData,layers,options);
%% Test accuracy for a several test sets

% For original training set
training_predictedLabels = classify(net,trainingData);
training_valLabels = trainingData.Labels;

training_accuracy = sum(training_predictedLabels == training_valLabels)/numel(training_valLabels)



