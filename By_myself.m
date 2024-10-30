clc; clear;

% 加载训练数据
trainData = imageDatastore('D:\Matlab_Project\CourseProject\BP\Datasets\train', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.jpg', ...
    'LabelSource', 'foldernames'); % 修改扩展名为 .jpg
testData = imageDatastore('Datasets/test', 'LabelSource', 'foldernames', 'FileExtensions', '.png');

imageSize = [32, 32];
numImages = numel(trainData.Files);

% 定义图像增强参数
imageAugmenter = imageDataAugmenter('RandRotation', [-20, 20], 'RandXReflection', true);

% 创建增强的图像数据存储
augmentedTrainData = augmentedImageDatastore(imageSize, trainData, 'DataAugmentation', imageAugmenter);
% 重置数据集指针到开头


% 读取增强后的图像
trainFeatures = zeros(numImages, prod(imageSize));


for i = 1:numImages

    img = readByIndex(augmentedTrainData, i); % 使用 readByIndex 按索引读取数据
    trainImg = cell2mat(img{1, 1}); % 获取图像数据
    Img = double(trainImg); % 转换为灰度图像并标准化到 [0, 1]
    train_img = (Img - min(Img(:))) / (max(Img(:)) - min(Img(:))); % 标准化到 [0, 1]
    trainFeatures(i, :) = train_img(:)'; % 将标准化后的图像展平并存储
end

% 初始化标签
trainLabels = zeros(numImages, 1);
for i = 1:numImages
    % 获取文件的路径并提取标签
    % [~, name, ~] = fileparts(trainData.Files{i});
    name = trainData.Labels(i);
    trainLabels(i) = double(name)-1;  % 将标签转换为数字，并加1以适应 MATLAB 的索引
end

% 生成一个随机的排列索引
randomIdx = randperm(numImages);

% 根据随机索引打乱训练数据和标签
shuffledFeatures = trainFeatures(randomIdx, :);
shuffledLabels = trainLabels(randomIdx, :);

% 拆分训练集和验证集
trainRatio = 0.8;
numTrainImages = round(numImages * trainRatio);

trainFeatures = shuffledFeatures(1:numTrainImages, :);
trainLabels = shuffledLabels(1:numTrainImages, :);% 假设 valFeatures 是验证集特征
valFeatures = shuffledFeatures(numTrainImages+1:end, :);%%验证集特征
valLabels = shuffledLabels(numTrainImages+1:end, :);


%%初始化神经网络
%%

% 初始化神经网络
inputLayer = size(trainFeatures, 2);
hiddenLayer1Size = 9;  % 第一个隐藏层节点数
hiddenLayer2Size = 1;  % 第二个隐藏层节点数
OutputLayerSize = 10;  % 对应0~9的10个标签

% 前馈权重和偏置初始化
W1 = randn(hiddenLayer1Size, inputLayer) * sqrt(2 / inputLayer); % He 初始化
b1 = zeros(hiddenLayer1Size, 1);  % 隐藏层偏置初始化为 0

W2 = randn(hiddenLayer2Size, hiddenLayer1Size) * sqrt(2 / hiddenLayer1Size); % He 初始化
b2 = zeros(hiddenLayer2Size, 1);  % 第二隐藏层偏置初始化为 0

W3 = randn(OutputLayerSize, hiddenLayer2Size) * sqrt(2 / hiddenLayer2Size); % He 初始化
b3 = zeros(OutputLayerSize, 1);   % 输出层偏置初始化为 0




batchSize = 20; % 批量学习的样本个数
valPredictions = zeros(size(valLabels)); % 初始化验证集的预测
numValBatches = ceil(size(valFeatures, 1) / batchSize); % 计算验证集批次数
totalValLoss = 0;

for batch = 1:numValBatches
    % 计算当前批次的索引
    startIdx = (batch - 1) * batchSize + 1;
    endIdx = min(batch * batchSize, size(valFeatures, 1));
    
    % 获取当前批次的特征
    X_val_batch = valFeatures(startIdx:endIdx, :)'; % 输入，维度为 [inputSize, batchSize]
    
    % 前向传播
    % 前向传播
    Z1_val = W1 * X_val_batch + b1; % 第一个隐藏层加权输入
    A1_val = activation(Z1_val, "sigmoid"); % 第一个隐藏层激活输出
    Z2_val = W2 * A1_val + b2; % 第二个隐藏层加权输入
    A2_val = activation(Z2_val, "sigmoid"); % 第二个隐藏层激活输出
    Z3_val = W3 * A2_val + b3; % 输出层加权输入
    A3_val = softmax(Z3_val); % 输出层激活输出


    
    % 计算交叉熵损失
    idx = sub2ind(size(A3_val), valLabels(startIdx:endIdx)' + 1, 1:size(A3_val, 2));
    batchLoss = -mean(log(A3_val(idx))); % 交叉熵损失计算
    
    % 累加批次损失到总损失
    totalValLoss = totalValLoss + batchLoss;
end

% 计算平均验证损失
averageValLoss = totalValLoss / numValBatches;
disp(['Initial Validation Loss: ', num2str(averageValLoss)]);



%%反向传播
%%
learningRate = 0.0001;
epochNum = 1000;
bestValLoss = inf;
numTrainImages = size(trainLabels,1);
batchSize = 20;
trainLosses = zeros(epochNum, 1); % 存储每个周期的训练损失
valLosses = zeros(epochNum, 1);    % 存储每个周期的验证损失
numBatches = ceil(numTrainImages / batchSize);
numBatches = size(numBatches,1);
%%
lambda = 0.019;  % 正则化强度

% Adam optimizer parameters
alpha = 0.001;  % 初始学习率
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;

% 初始化参数
mW1 = zeros(size(W1)); vW1 = zeros(size(W1));
mb1 = zeros(size(b1)); vb1 = zeros(size(b1));
mW2 = zeros(size(W2)); vW2 = zeros(size(W2));
mb2 = zeros(size(b2)); vb2 = zeros(size(b2));
mW3 = zeros(size(W3)); vW3 = zeros(size(W3));
mb3 = zeros(size(b3)); vb3 = zeros(size(b3));



% 反向传播
for epoch = 1:epochNum
    epochLoss = 0;
    
    for batch = 1:numBatches
        % 计算当前批次的索引
        startIdx = (batch - 1) * batchSize + 1;
        endIdx = min(batch * batchSize, numTrainImages);
        
        % 获取当前批次的数据
        X_batch = trainFeatures(startIdx:endIdx, :)';  % 输入
        Y_batch = trainLabels(startIdx:endIdx)';    % 标签
        
        % 前向传播
        Z1 = W1 * X_batch + b1;          % 第一个隐藏层加权输入
        A1 = activation(Z1, "sigmoid");  % 第一个隐藏层激活输出
        Z2 = W2 * A1 + b2;               % 第二个隐藏层加权输入
        A2 = activation(Z2, "sigmoid");  % 第二个隐藏层激活输出
        Z3 = W3 * A2 + b3;               % 输出层加权输入
        A3 = softmax(Z3);                % 输出层激活输出
        
        % 计算误差
        idx = sub2ind(size(A3), Y_batch + 1, 1:size(A3, 2));
        epsilon = 1e-10;
        error = -mean(log(A3(idx) + epsilon));  % 使用平均交叉熵
        epochLoss = epochLoss + error;
        
        % 反向传播
        delta3 = A3;  % 初始化 delta3 为 A3 的概率
        delta3(sub2ind(size(A3), Y_batch + 1, 1:size(A3, 2))) = delta3(sub2ind(size(A3), Y_batch + 1, 1:size(A3, 2))) - 1;
        delta2 = (W3' * delta3) .* diff_activation(A2, "sigmoid");  % 第二隐藏层误差项
        delta1 = (W2' * delta2) .* diff_activation(A1, "sigmoid");  % 第一个隐藏层误差项
        
        % 计算梯度
        gradW3 = delta3 * A2' / size(A3, 2) + lambda * W3; % 加入 L2 正则化
        gradb3 = mean(delta3, 2);
        gradW2 = delta2 * A1' / size(A3, 2) + lambda * W2; % 加入 L2 正则化
        gradb2 = mean(delta2, 2);
        gradW1 = delta1 * X_batch' / size(A3, 2) + lambda * W1; % 加入 L2 正则化
        gradb1 = mean(delta1, 2);
        
        % 更新一阶矩估计
        mW1 = beta1 * mW1 + (1 - beta1) * gradW1;
        mb1 = beta1 * mb1 + (1 - beta1) * gradb1;
        mW2 = beta1 * mW2 + (1 - beta1) * gradW2;
        mb2 = beta1 * mb2 + (1 - beta1) * gradb2;
        mW3 = beta1 * mW3 + (1 - beta1) * gradW3;
        mb3 = beta1 * mb3 + (1 - beta1) * gradb3;
        
        % 更新二阶矩估计
        vW1 = beta2 * vW1 + (1 - beta2) * (gradW1 .^ 2);
        vb1 = beta2 * vb1 + (1 - beta2) * (gradb1 .^ 2);
        vW2 = beta2 * vW2 + (1 - beta2) * (gradW2 .^ 2);
        vb2 = beta2 * vb2 + (1 - beta2) * (gradb2 .^ 2);
        vW3 = beta2 * vW3 + (1 - beta2) * (gradW3 .^ 2);
        vb3 = beta2 * vb3 + (1 - beta2) * (gradb3 .^ 2);
        
        % 计算偏差校正后的估计
        mW1_hat = mW1 / (1 - beta1^epoch);
        mb1_hat = mb1 / (1 - beta1^epoch);
        mW2_hat = mW2 / (1 - beta1^epoch);
        mb2_hat = mb2 / (1 - beta1^epoch);
        mW3_hat = mW3 / (1 - beta1^epoch);
        mb3_hat = mb3 / (1 - beta1^epoch);
        vW1_hat = vW1 / (1 - beta2^epoch);
        vb1_hat = vb1 / (1 - beta2^epoch);
        vW2_hat = vW2 / (1 - beta2^epoch);
        vb2_hat = vb2 / (1 - beta2^epoch);
        vW3_hat = vW3 / (1 - beta2^epoch);
        vb3_hat = vb3 / (1 - beta2^epoch);
        
        % 更新权重和偏置
        W1 = W1 - alpha * mW1_hat ./ (sqrt(vW1_hat) + epsilon);
        b1 = b1 - alpha * mb1_hat ./ (sqrt(vb1_hat) + epsilon);
        W2 = W2 - alpha * mW2_hat ./ (sqrt(vW2_hat) + epsilon);
        b2 = b2 - alpha * mb2_hat ./ (sqrt(vb2_hat) + epsilon);
        W3 = W3 - alpha * mW3_hat ./ (sqrt(vW3_hat) + epsilon);
        b3 = b3 - alpha * mb3_hat ./ (sqrt(vb3_hat) + epsilon);
    end
    
    % 计算训练损失，加入 L2 正则化
    trainLosses(epoch) = (epochLoss / numBatches) + (lambda / 2) * (norm(W1, 'fro')^2 + norm(W2, 'fro')^2 + norm(W3, 'fro')^2);
    
    % 验证损失计算（无 Dropout）
    valPredictions = zeros(size(valLabels));
    
    % 验证损失计算（无循环）
    Z1_val = W1 * valFeatures' + b1;
    A1_val = activation(Z1_val, "sigmoid");
    Z2_val = W2 * A1_val + b2;
    A2_val = activation(Z2_val, "sigmoid");
    Z3_val = W3 * A2_val + b3;
    A3_val = softmax(Z3_val);
    
    % 选择每个样本的实际类别概率作为验证损失的计算基础
    valIdx = sub2ind(size(A3_val), valLabels' + 1, 1:size(A3_val, 2));
    valLoss = -mean(log(A3_val(valIdx) + epsilon));  % 计算平均交叉熵损失
    valLoss = valLoss + (lambda / 2) * (norm(W1, 'fro')^2 + norm(W2, 'fro')^2 + norm(W3, 'fro')^2);  % 加入验证损失的正则化
    valLosses(epoch) = valLoss;
    
    % 预测类别
    [~, predictedLabels] = max(A3_val, [], 1); 
    valPredictions = predictedLabels - 1; % 将索引减1以匹配标签范围0-9
    
    disp(['Epoch: ' num2str(epoch) ', Training Error: ' num2str(epochLoss / numBatches) ', Validation Loss: ' num2str(valLoss)]);
end



% 绘制训练损失和验证损失曲线
figure;
hold on;
plot(trainLosses(1:epoch), 'LineWidth', 1.5);
plot(valLosses(1:epoch), 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('Loss');
title('Training and Validation Loss');
legend('Training Loss', 'Validation Loss');
grid on;
hold off;

% 测试集的预测（无 Dropout）
%%%简单测试
testFeatures = zeros(numel(testData.Files), prod(imageSize));
for i = 1:numel(testData.Files)
    img = imread(testData.Files{i}); % 使用测试数据集文件
    img = imresize(rgb2gray(im2double(img)), imageSize); % 归一化和尺寸调整
    img = (img - mean(img(:))) / std(img(:)); % 标准化到零均值和单位方差
    testFeatures(i, :) = img(:)'; % 展平图像并存储
end
testPredictions = zeros(size(testFeatures, 1), OutputLayerSize);
for j = 1:size(testFeatures, 1)
    X_test = testFeatures(j, :)';  % 获取单个测试样本特征向量
    
    % 前向传播（无 Dropout）
    Z1_test = W1 * X_test + b1;
    A1_test = activation(Z1_test, "sigmoid");
    Z2_test = W2 * A1_test + b2;
    A2_test = activation(Z2_test, "sigmoid");  % 第二隐藏层激活输出
    Z3_test = W3 * A2_test + b3;
    testPredictions(j, :) = softmax(Z3_test)';  % 最终输出为 softmax 概率
end

% 获取预测类别（最大值对应的索引）
[~, predictedLabels] = max(testPredictions, [], 2);
numericLabels = [];
for i =1:10
    numericLabels(i)=i-1;
end
numericLabels=numericLabels';
% 显示测试集的预测结果
contrast = [numericLabels, predictedLabels - 1]
correctPredictions = sum(numericLabels == (predictedLabels - 1));
accuracy = correctPredictions / length(numericLabels);
disp(['Accuracy: ' num2str(accuracy)]);


