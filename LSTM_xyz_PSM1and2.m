% Parinaz , Read Rosbag and load sequence data
clc; close all; clearvars -except psmData;
% read bag files pattern 2
bag_file_joy_2 = rosbag('subject2_processed/4_joystick_pattern2_sim_processd.bag');
bag_file_subject2_Auto_2 = rosbag('subject2_processed/8_Autocamera_pattern2_sim_processd.bag');
bag_file_subject2_clutch_2 = rosbag('subject2_processed/5_clutch_control_pattern2_sim_processd.bag');
% select the topic
topic_PSM1_joy_2 = bag_file_joy_2.select('Topic', '/dvrk/PSM1/pose_current');
topic_PSM1_Auto_2 = bag_file_subject2_Auto_2.select('Topic', '/dvrk/PSM1/pose_current');
topic_PSM1_clutch_2 = bag_file_subject2_clutch_2.select('Topic', '/dvrk/PSM1/pose_current');
topic_PSM2_joy_2 = bag_file_joy_2.select('Topic', '/dvrk/PSM2/pose_current');
topic_PSM2_Auto_2 = bag_file_subject2_Auto_2.select('Topic', '/dvrk/PSM2/pose_current');
topic_PSM2_clutch_2 = bag_file_subject2_clutch_2.select('Topic', '/dvrk/PSM2/pose_current');
% read message of selected topics
data_PSM1_joy_2 = readMessages(topic_PSM1_joy_2);
data_PSM1_Auto_2 = readMessages(topic_PSM1_Auto_2);
data_PSM1_clutch_2 = readMessages(topic_PSM1_clutch_2);
data_PSM2_joy_2 = readMessages(topic_PSM2_joy_2);
data_PSM2_Auto_2 = readMessages(topic_PSM2_Auto_2);
data_PSM2_clutch_2 = readMessages(topic_PSM2_clutch_2);
% concatenate data
data_PSM1 = [data_PSM1_joy_2; data_PSM1_Auto_2 ; data_PSM1_clutch_2];
data_PSM2 = [data_PSM2_joy_2; data_PSM2_Auto_2 ; data_PSM2_clutch_2];
for i = 1:(numel(data_PSM1))
    px1 = data_PSM1{i}.Position.X;
    py1 = data_PSM1{i}.Position.Y;
    pz1 = data_PSM1{i}.Position.Z;
    ox1 = data_PSM1{i}.Orientation.X;
    oy1 = data_PSM1{i}.Orientation.Y;
    oz1 = data_PSM1{i}.Orientation.Z;
    px2 = data_PSM2{i}.Position.X;
    py2 = data_PSM2{i}.Position.Y;
    pz2 = data_PSM2{i}.Position.Z;
    ox2 = data_PSM2{i}.Orientation.X;
    oy2 = data_PSM2{i}.Orientation.Y;
    oz2 = data_PSM2{i}.Orientation.Z;
     
    psmData1{i} = [px1;ox1;py1;oy1;pz1;oz1];
    psmData2{i} = [px2;ox2;py2;oy2;pz2;oz2];
end
psmData1_mat = cell2mat(psmData1);
psmData2_mat = cell2mat(psmData2);
psmData = [psmaData1_mat; psmData2_mat];
OutputWindowSize = 50;
InputWindowSize = 100;
numTrainingData = floor(0.9*size(psmData,2)) - (OutputWindowSize + InputWindowSize);
numData = size(psmData,2) - (OutputWindowSize + InputWindowSize);

for i = 1:numData
    InputData(i,:,:) = psmData(:,i:i+InputWindowSize -1);
    OutputData(i,:,:) = psmData(:,i+InputWindowSize:OutputWindowSize+i+InputWindowSize-1);
end
InputTrainData = reshape(InputData,numData, InputWindowSize* 12);
TargetTrainData = reshape(OutputData, numData, OutputWindowSize* 12);
% standardize data
%mu = mean(TrainData);
%sigma = std(TrainData);
%TrainDataStandard = (TrainData - mu) / sigma;
% divide data to input and output dataset
% TrainDataStandard = TrainData;
% Data = reshape(TrainDataStandard(1:(input_length+prediction_length)*numTrainData),numTrainData,input_length+prediction_length);
% InputData = Data(:,1:input_length);
% OutputData = Data(:,input_length+1:end);
% Define LSTM
numFeatures = InputWindowSize*12;
numOutputs = OutputWindowSize*12;
numHiddenUnits_1 = 400;
numHiddenUnits_2 = 200;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits_1)
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits_2)
    dropoutLayer(0.2)
    fullyConnectedLayer(numOutputs)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'MiniBatchSize',100, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.1, ...
    'Verbose',0, ...
    'Plots','training-progress');
net = trainNetwork(InputData',OutputData',layers,options);
%% forcasting
%dataTestStandardized = (TestData - mu) / sigma;
%XTest = dataTestStandardized(1:end-1);
DataTest = reshape(TestData(1:(InputWindowSize+OutputWindowSize)*numTestData),numTestData,InputWindowSize+OutputWindowSize);
InputTest = (DataTest(:,1:InputWindowSize))';
OutputTest = (DataTest(:,InputWindowSize + 1:end))';
net = predictAndUpdateState(net,InputData');
[net,YPred] = predictAndUpdateState(net,InputTest(:,1));
numTimeStepsTest = size(DataTest,1);
[net,YPred_1] = predictAndUpdateState(net,[InputTest(1:OutputWindowSize,1);YPred]);

rmse = sqrt(mean((YPred-OutputTest(:,1)).^2));

figure
subplot(2,1,1)
plot([OutputTest(:,1);OutputTest(:,2)])
hold on
plot([YPred;YPred_1],'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred - OutputTest)
xlabel("Month")
ylabel("Error")
%title("RMSE = " + rmse)







