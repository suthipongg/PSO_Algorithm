function fitness_error = TestFunction(x, FuncNo)


[m, n] = size(x);

switch FuncNo
    
    % Ackley 1 Function
    case 1
        ninverse = 1 / n;
        sum1 = sum(x .^ 2, 2);
        sum2 = sum(cos(2 * pi * x), 2);
        scores = 20 + exp(1) - (20 * exp(-0.2 * sqrt( ninverse * sum1))) - exp( ninverse * sum2);

    case 2
        scores = (sin(3*pi*x(:, 1)))^2 + ((x(:, 1) - 1)^2)*(1 + (sin(3*pi*x(:, 2)))^2) + ((x(:, 2) - 1)^2)*(1 + (sin(2*pi*x(:, 2)))^2);

    case 3
        scores = -(x(:, 1) + 47)*(sin(sqrt(abs((x(:, 2)/2) + (x(:, 1) + 47))))) - (x(:, 2))*(sin(sqrt(abs(x(:, 2) - (x(:, 1) + 47))))) + 959.6407;

    case 4
        scores = -(abs((sin(x(:, 1))*(cos(x(:, 2)))*(exp(abs(1 - ((sqrt(x(:, 1)^2 + x(:, 2)^2))/pi))))))) + 19.2085;
    % Rastrigin Function
    case 5
        A = 10;
        scores = (A * n) + (sum(x .^2 - A * cos(2 * pi * x), 2));

    case 6
        sum1 = sum((x .*sin(sqrt(abs(x)))), 2);
        scores = 418.9829*n + sum1;

          
end
% switch end

% ่ทๅ–ๆต่ฏ•ๅฝๆ•ฐๆ?ผๅ€?
[~, ~, opt_f] = Get_Func_Info(n, FuncNo);

% ่ฎก็ฎ—่ฏฏๅทฎๅ€?
fitness_error = abs(scores - opt_f);

end