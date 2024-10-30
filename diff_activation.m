function y = diff_activation(x, method)
    % 激活函数梯度: sigmoid, tanh, relu, softmax
    if strcmp(method, "sigmoid")
        y = (x) .* (1 - x);  % Sigmoid 的导数
    elseif strcmp(method, "tanh")
        y = 1 - (x).^2;      % Tanh 的导数
    elseif strcmp(method, "relu")
        y = x > 0;           % ReLU 的导数: 大于0的部分为1，其他部分为0
    elseif strcmp(method, "softmax")
        % Softmax 的导数: 对角线部分
        s = exp(x) ./ sum(exp(x));
        y = diag(s) - (s * s');
    else
        error('Unsupported activation method. Choose from: sigmoid, tanh, relu, softmax.');
    end
end
