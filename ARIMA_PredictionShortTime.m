clc; clear all;close all;
subject1pospsm1 = readtable('C:\Users\parinaz\OneDrive\subject1_processed\subject1_pos_psm1.csv');
len = length(subject1pospsm1.dvrk_PSM1_pose_current__position_x);
test_length = 500;
train_length = 500;
training_data_x = subject1pospsm1.dvrk_PSM1_pose_current__position_x(1:train_length/10);
test_data_x = subject1pospsm1.dvrk_PSM1_pose_current__position_x(train_length/10+1:train_length/10+test_length/10);
x_train = 1:1:train_length/10;
x_test = train_length/10+1:train_length/10+test_length/10;
% log
PSM1Log_x = log(training_data_x);
% Remove Linear 
CloseLogLinear_x = detrend(PSM1Log_x);
% Differences 
CloseLogLinearDifferences_x = diff(CloseLogLinear_x);
Differences_x = diff(training_data_x);
% View the data
figure
plot(x_train, training_data_x);
hold on
plot(x_train, PSM1Log_x);
plot(x_train, CloseLogLinear_x);
plot(x_train(2:end), CloseLogLinearDifferences_x);
plot(x_train(2:end), Differences_x);
title('PSM X Training');
ylabel('Position')
xlabel('Time(10ms)')
legend('original x pos', '1.log', '2.DetrendLinear', '3.1st order Differences','4.1st order Differences of pos x', 'Location' , 'Best')
grid on
hold off

% Test Stationary 
[h,p] = adftest(CloseLogLinearDifferences_x);
display(h)
[h1,p1] = adftest(Differences_x);
display(h1)
% Autocorrelation function (ACF)
figure
autocorr(Differences_x,'NumLags', 20, 'NumMA',1)
% Partial ACF (PACF)
figure
parcorr(Differences_x, 'NumAR', 1, 'NumLags', 20);
%ARIMA model
ARIMA_model_x_pos = arima(2,0,1);
[ARIMA_model_x_pos1,~,Loglikehood] = estimate(ARIMA_model_x_pos,training_data_x);
% Prediction Results
rng(1); % For reproducibility
%View the predictive vallue
residual = infer(ARIMA_model_x_pos1,training_data_x);
figure
autocorr(residual, 'NumLags', 20, 'NumMA', 1)
prediction = training_data_x + residual;
figure
plot(x_train,training_data_x);
hold on
plot(x_train,prediction);
title('PSM X ARIMA Result');
ylabel('Position')
xlabel('Time(10ms)')
legend('Original','ARIMA(2,0,1)','Location','best');
grid on
hold off
%%
% see the residual
histogram(residual)
% Find the best ARIMA
[aic,bic] = aicbic(Loglikehood,2,250);
Name = "Arima(0,1,0)";
AIC = aic;
BIC = bic;
W=2;
for i = 1:2
    for j = 1:2
        for k=1:2
            ARIMA_CloseE(W-1) = arima(i,j,k);
            [~,~,LoglikehoodE] = estimate(ARIMA_CloseE(W-1),subject1pospsm1.dvrk_PSM1_pose_current__position_x,'display','off');
            [aicE,bicE] = aicbic(LoglikehoodE,2,250);
            a=convertCharsToStrings(strcat('Arima(',num2str(i),',',num2str(j),',',num2str(k),')'));
            Name(W) = a;
            AIC(W) = aicE;
            BIC(W) = bicE;
            W=W+1;
        end
    end
end
TableComparison = table;
TableComparison = table(Name',AIC',BIC');
TableComparison.Properties.VariableNames = {'NameModel','AIC','BIC'};
%find the lowest AIC and BIC
AICWin = AIC(1);
recordAIC = i;
%Lowest AIC
for i=1:length(TableComparison.AIC)
    if TableComparison.AIC(i)>AICWin
        recordAIC=i;
    end
end
BICWin = BIC(1);
recordBIC = i;
%Lowest BIC
for i=1:length(TableComparison.BIC)
    if TableComparison.BIC(i)>BICWin
        recordBIC=i;
    end
end

%The lowest AIC and BIC is model ARIMA(2,2,2)
%Hence we use ARIMA(2,2,2) for forecastinh
ARIMA_CloseFinal = arima(2,2,2);
[ARIMA_CloseFinal,~,LoglikehoodE] = estimate(ARIMA_CloseFinal,training_data_x,'display','off');
rng(1); % For reproducibility
% View the predictive value
residual = infer(ARIMA_CloseFinal,training_data_x);
prediction = training_data_x + residual;
figure
plot(x_train,training_data_x);
hold on
plot(x_train,prediction);
title('PSM1 X Position');
ylabel('Position')
xlabel('Time(10ms)')
legend('Original','ARIMA(2,2,2)','Location','best');
grid on
hold off

%% Test model
ARIMA_model_second_candidate = arima(1,1,1);
[ARIMA_model_second_candidate,~,Loglikehood] = estimate(ARIMA_model_second_candidate,training_data_x);

ARIMA_CloseFinalForecast1 = forecast(ARIMA_CloseFinal,test_length/10,'Y0',test_data_x);
ARIMA_CloseFinalForecast2 = forecast(ARIMA_model_second_candidate,test_length/10,'Y0',test_data_x);
figure
plot(x_test,test_data_x);
hold on
plot(x_test, ARIMA_CloseFinalForecast1);
plot(x_test, ARIMA_CloseFinalForecast2);
title('PSM1 X Position');
ylabel('Position')
xlabel('Time(10ms)')
legend('OriginalPSM1','ARIMA(2,2,2)','ARIMA(1,1,1)','Location','best');
grid on
hold off
%Use Monte Carlo
[Montey,k]=simulate(ARIMA_CloseFinal,test_length/10,'NumPath',100,'Y0',training_data_x);
figure
plot(x_train,training_data_x);
hold on
plot(x_test, test_data_x,'b:');
plot(x_test,Montey);
title('PSM1 X Position Monte Carlo Prediction');
ylabel('Position')
xlabel('Time(10ms)')
legend('OriginalPSM1','Location','best');
avg_prediction = mean(Montey, 2);
error_prediction = sqrt((test_data_x - avg_prediction).^2);
figure
plot(x_test, error_prediction);
title('PSM1 X Position Mean Squre Error Monte Carlo');
ylabel('Position')
xlabel('Time(10ms)')
legend('error','Location','best');
figure
plot(x_test, var(Montey, 0, 2));
title('Monte Carlo Variance');
ylabel('Position')
xlabel('Time(10ms)')
legend('Variance','Location','best');