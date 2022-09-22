%% read in the files

RCDatasetPath = fullfile(cd,'rightCover_test');
RCTestData = imageDatastore(RCDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% Load the trained network
load('net.mat');

%% Test accuracy for a several test sets

RC_predictedLabels = classify(net,RCTestData);
RC_valLabels = RCTestData.Labels;

RC_accuracy = sum(RC_predictedLabels == RC_valLabels)/numel(RC_valLabels)



