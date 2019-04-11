% Parinaz , Read Rosbag and load sequence data
clc; close all; clear all;
bag_file = rosbag('subject2_processed/4_joystick_pattern2_sim_processd.bag');
topic_PSM1 = bag_file.select('Topic', '/dvrk/PSM1/pose_current');
topic_PSM2 = bag_file.select('Topic', '/dvrk/PSM2/pose_current');
data_PSM1 = readMessages(topic_PSM1);
data_PSM2 = readMessages(topic_PSM2);
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
%     
%     psmData1{i} = [px1;py1;pz1];
%     psmData2{i} = [px2;py2;pz2];
     psmData1{i} = [px1;ox1;py1;oy1;pz1;oz1];
     psmData2{i} = [px2;ox2;py2;oy2;pz2;oz2];
end
psmData1_mat = cell2mat(psmData1);
psmData2_mat = cell2mat(psmData2);
figure
for i = 1:size(psmData1_mat,1)
    plot(psmData1_mat(i,:))
    hold on
end
legend('position_x', 'orientation_x', 'position_y', 'orientation_y', 'position_z','orientation_z');
% create input sequence for LSTM, divide data to time and test dataset 
Data = [psmData1_mat; psmData2_mat];
numTrainingData = floor(0.9*size(Data,2));
TrainData = Data(:,1:numTrainingData+1);
TestData = Data(:,numTrainingData+1:end);

% standardize data
mu = mean(TrainData,2);
sigma = std(TrainData,0,2);
TrainDataStandard = (TrainData - mu) ./ sigma;
% divide data to input and output dataset
%prediction_length = 500;
InputData = TrainData(:,1:end - 1);
OutputData = TrainData(:,2: end);
% Define LSTM
numFeatures = 12;
numOutputs = 12;
numHiddenUnits = 200;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numOutputs)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
net = trainNetwork(InputData,OutputData,layers,options);
%% forcasting
dataTestStandardized = (TestData - mu) ./ sigma;
XTest = dataTestStandardized(:,1:end-1);
net = predictAndUpdateState(net,InputData);
[net,YPred] = predictAndUpdateState(net,OutputData(:,end));
numTimeStepsTest = size(XTest,2);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end
YPred = sigma.*YPred + mu;
YTest = TestData(:,2:end);
rmse = sqrt(mean((YPred-YTest).^2));
% figure
% plot(TrainData(1,1:end-1))
% hold on
% idx = numTrainingData:(numTrainingData+numTimeStepsTest);
% 
% plot(idx,[Data(:,numTrainingData) YPred],'.-')
% hold off
% xlabel("Month")
% ylabel("Cases")
% title("Forecast")
% legend(["Observed" "Forecast"])
figure
subplot(2,1,1)
plot(YTest(1,:))
hold on
plot(YPred(1,:),'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred(1,:) - YTest(1,:))
xlabel("Month")
ylabel("Error")
%title("RMSE = " + rmse)







