clear;
%rng('shuffle');

%loading dataset, the cost of train and validate are averaged by 5 times
% the cost of testing are not averaged
load data_train.mat;
load cost_train.mat;
load data_validate;
load cost_validate;
load data_test.mat;
load cost_test.mat;
% the 104 dimension for data is encoded as described below:
% 1:52 is the hand of player 1, 53:104 is the hand of player 2,
% for the representation of the 52 dimensions for each player
% it is a 13-hot encoding from [S2, S3, S4, ... SA, H2, H3,
% H4, ..., HA, D2, D3, D4, ..., DA, C2, C3, C4, ... CA],
% where 1 implies the player has that card, and 0 otherwise

data_insample = data;
cost_insample = cost;
% get the feature of the dataset, feature{1} is for player 1, feature{2} is
% for player 2, and the 5 dimensions are respectively the number of cards
% for spade,heart,diamond,clubs and the high-card points
[feature, feature_v] = get_feature(data,data_validate);

%decide the maximum bidding length and the batchsize for validation
% choose totalbid from 2 to 5
totalbid =4;
batchsizev = 1;

% training parameters
update_dnntype = 2;
badupdate_dnn = 2;
explore_first = 1;
alphaupdate_dnn = 0.1;
batchsizeupdate_dnn = 50;
batchsize = 1;
decayRate = 0.98;
momentum = 0.82;
alpha = 0.83;
startbackprop = 0;
input = 52+36+5;
lsize = 128;
layer = 4;
output = 36;
eta = 0.05;

% initialization for the model and RMSprop parameters
WW_qlearning = cell(1,totalbid);
BB_qlearning = cell(1,totalbid);
dW_qlearning = cell(1,totalbid);
dB_qlearning = cell(1,totalbid);
sW_qlearning = cell(1,totalbid);
sB_qlearning = cell(1,totalbid);
[WW_qlearning{1}, BB_qlearning{1}, dW_qlearning{1}, dB_qlearning{1}] = init_nogpu(52, output, lsize, layer);
[sW_qlearning{1}, sB_qlearning{1}] = sinit_nogpu([data(1:52,:)], cost, WW_qlearning{1}, BB_qlearning{1}, output);
for bid = 2:2:totalbid
    [WW_qlearning{bid}, BB_qlearning{bid}, dW_qlearning{bid}, dB_qlearning{bid}] = init_nogpu(input, output, lsize, layer);
    [sW_qlearning{bid}, sB_qlearning{bid}] = sinit_nogpu([data(53:104,:);ones(41,size(data,2))], cost, WW_qlearning{bid}, BB_qlearning{bid}, output);
end
for bid = 3:2:totalbid
    [WW_qlearning{bid}, BB_qlearning{bid}, dW_qlearning{bid}, dB_qlearning{bid}] = init_nogpu(input, output, lsize, layer);
    [sW_qlearning{bid}, sB_qlearning{bid}] = sinit_nogpu([data(1:52,:);ones(41,size(data,2))], cost, WW_qlearning{bid}, BB_qlearning{bid}, output);
end

%load a trained model for total bid is 2 to 5, overrides the WW_qlearningn
%and BB_qlearning
if totalbid ==2
    load('model_valcost_1.152836e-01_totalbid2_4_128_50_3.692846e-02_9.800000e-01_8.200000e-01_alpha1.000000e-01.mat');
elseif totalbid ==3
    load('model_valcost_1.085400e-01_totalbid3_4_128_50_2.515687e-02_9.800000e-01_8.200000e-01_alpha1.000000e-01.mat');
elseif totalbid ==4
    load('model_valcost_1.063088e-01_totalbid4_4_128_50_3.204176e-03_9.800000e-01_8.200000e-01_alpha1.000000e-01.mat');
elseif totalbid==5
    load('model_valcost_1.076380e-01_totalbid5_4_128_50_2.955429e-03_9.800000e-01_8.200000e-01_alpha5.000000e-02.mat');
end
% training_result
%heckError_qlearning_103_insample;
% validation result, the comments for the cost-calculating code are in checkError_qlearning_103
checkError_qlearning_103;
% test result
%checkError_qlearning_103_testing;
