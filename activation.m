function y = activation(x, method)
    %% 可选项: sigmoid, tanh, relu, softmax
    if strcmp(method, "sigmoid")
        y = logsig(x);  % Sigmoid 激活函数
    elseif strcmp(method, "tanh")
        y = tanh(x);    % Tanh 激活函数
    elseif strcmp(method, "relu")
        y = max(0, x);  % ReLU 激活函数
    elseif strcmp(method, "softmax")
        % expX = exp(x - max(x)); % 为数值稳定性进行调整
        % y = expX / sum(expX);   % Softmax 激活函数
        y = softmax(x);
    else
        error('Unsupported activation method. Choose from: sigmoid, tanh, relu, softmax.');
    end
end
