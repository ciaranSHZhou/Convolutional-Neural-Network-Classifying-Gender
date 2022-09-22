%% read in the files
%Training dataset
trainingDatasetPath = fullfile(cd,'dataset');
trainingData = imageDatastore(trainingDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%mirror test dataset
mirrorTestsetPath = fullfile(cd,'mirror_test');
mirrorTestData = imageDatastore(mirrorTestsetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%Upside-down test dataset
USDTestsetPath = fullfile(cd,'upsidedown_test');
USDTestData = imageDatastore(USDTestsetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%Upper cover test dataset
UCTestsetPath=fullfile(cd,'upperCover_test');
UCTestData = imageDatastore(UCTestsetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%Lower cover test dataset
LCTestsetPath=fullfile(cd,'lowerCover_test');
LCTestData = imageDatastore(LCTestsetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% 10%/ 50%/ 80%/ 90%/ 100% noisy test dataset
PCT10TestSetPath=fullfile(cd,'10pct_noisy_test');
PCT10TestData = imageDatastore(PCT10TestSetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

PCT50TestSetPath=fullfile(cd,'50pct_noisy_test');
PCT50TestData = imageDatastore(PCT50TestSetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');


PCT80TestSetPath=fullfile(cd,'80pct_noisy_test');
PCT80TestData = imageDatastore(PCT80TestSetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');


PCT90TestSetPath=fullfile(cd,'90pct_noisy_test');
PCT90TestData = imageDatastore(PCT90TestSetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');


PCT100TestSetPath=fullfile(cd,'100pct_noisy_test');
PCT100TestData = imageDatastore(PCT100TestSetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%Novel faces test data
novelTestSetPath=fullfile(cd,'novel_test');
novelTestData = imageDatastore(novelTestSetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%% Load the trained network
load('net.mat');

%% Test accuracy for a several test sets
% For original training set
training_predictedLabels = classify(net,trainingData);
training_valLabels = trainingData.Labels;

training_accuracy = sum(training_predictedLabels == training_valLabels)/numel(training_valLabels)

% For mirror test set
mirror_predictedLabels = classify(net,mirrorTestData);
mirror_valLabels = mirrorTestData.Labels;

mirror_accuracy = sum(mirror_predictedLabels == mirror_valLabels)/numel(mirror_valLabels)

% For upside-down test set
USD_predictedLabels = classify(net,USDTestData);
USD_valLabels = USDTestData.Labels;

USD_accuracy = sum(USD_predictedLabels == USD_valLabels)/numel(USD_valLabels)


% For upper face cover test set
UC_predictedLabels = classify(net,UCTestData);
UC_valLabels = UCTestData.Labels;

UC_accuracy = sum(UC_predictedLabels == UC_valLabels)/numel(UC_valLabels)

% For lower face cover test set
LC_predictedLabels = classify(net,LCTestData);
LC_valLabels = LCTestData.Labels;

LC_accuracy = sum(LC_predictedLabels == LC_valLabels)/numel(LC_valLabels)

% For noisy test set
PCT10_predictedLabels = classify(net,PCT10TestData);
PCT10_valLabels = PCT10TestData.Labels;
PCT10_accuracy = sum(PCT10_predictedLabels == PCT10_valLabels)/numel(PCT10_valLabels)

PCT50_predictedLabels = classify(net,PCT50TestData);
PCT50_valLabels = PCT50TestData.Labels;
PCT50_accuracy = sum(PCT50_predictedLabels == PCT50_valLabels)/numel(PCT50_valLabels)

PCT80_predictedLabels = classify(net,PCT80TestData);
PCT80_valLabels = PCT80TestData.Labels;
PCT80_accuracy = sum(PCT80_predictedLabels == PCT80_valLabels)/numel(PCT80_valLabels)

PCT90_predictedLabels = classify(net,PCT90TestData);
PCT90_valLabels = PCT90TestData.Labels;
PCT90_accuracy = sum(PCT90_predictedLabels == PCT90_valLabels)/numel(PCT90_valLabels)

PCT100_predictedLabels = classify(net,PCT100TestData);
PCT100_valLabels = PCT100TestData.Labels;
PCT100_accuracy = sum(PCT100_predictedLabels == PCT100_valLabels)/numel(PCT100_valLabels)

% Test novel faces
novel_predictedLabels = classify(net,novelTestData);
novel_valLabels = novelTestData.Labels;
novel_accuracy = sum(novel_predictedLabels == novel_valLabels)/numel(novel_valLabels)
