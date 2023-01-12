function fitness_error = TestFunction(x, FuncNo)


[m, n] = size(x);

switch FuncNo
    
    % Brown Function
    case 1
        scores = 0;
        x = x .^ 2;
        for i = 1:(n-1)
            scores = scores + x(:, i) .^ (x(:, i+1) + 1) + x(:, i+1).^(x(:, i) + 1);
        end
    
    % Exponential Function
    case 2
        x2 = x .^ 2;
        scores = -exp(-0.5 * sum(x2, 2));
    
    % Griewank Function: 
    case 3 
        sumcomp = 0;
        prodcomp = 1;

        for i = 1:n
            sumcomp = sumcomp + (x(:, i) .^ 2);
            prodcomp = prodcomp .* (cos(x(:, i) / sqrt(i)));
        end

        scores = (sumcomp / 4000) - prodcomp + 1;
    
   
  
    % Ridge Function
    case 4
        d = 1;
        alpha = 0.5;
        x1 = x(:, 1);
        scores = x1 + d * (sum(x(:, 2:end).^2, 2) .^ alpha);        

    % Schwefel 2.20 Function
    case 5
        scores = sum(abs(x), 2);
    
    % Schwefel 2.21 Function
    case 6
        scores = max(abs(x), [], 2);
    
    % Schwefel 2.22 Function
    case 7
        absx = abs(x);
        scores = sum(absx, 2) + prod(absx, 2);
    
    % Schwefel 2.23 Function
    case 8
        scores = sum(x .^10, 2);
    
    % Sphere Function
    case 9
        scores = sum(x .^ 2, 2);
    
    % Sum Squares Function
    case 10
       x2 = x .^2;
       I = repmat(1:n, m, 1);
       scores = sum( I .* x2, 2);
    
    % Xin-She Yang (Function 3)
    case 11
        m1 = 5;
        beta = 15;
        scores = exp(-sum((x / beta).^(2*m1), 2)) - (2 * exp(-sum(x .^ 2, 2)) .* prod(cos(x) .^ 2, 2));

    % Zakharov Function
    case 12
        comp1 = 0;
        comp2 = 0;
        for i = 1:n
            comp1 = comp1 + (x(:, i) .^ 2);
            comp2 = comp2 + (0.5 * i * x(:, i));
        end
        scores = comp1 + (comp2 .^ 2) + (comp2 .^ 4);
    
    % Ackley 1 Function
    case 13
        ninverse = 1 / n;
        sum1 = sum(x .^ 2, 2);
        sum2 = sum(cos(2 * pi * x), 2);
        scores = 20 + exp(1) - (20 * exp(-0.2 * sqrt( ninverse * sum1))) - exp( ninverse * sum2);
    
    
    
    % Alpine 1 Function
    case 14
        scores = sum(abs(x .* sin(x) + 0.1 * x), 2);
    
   
    
    % Happy Cat  function
    case 15
        alpha = 0.5;
        x2 = sum(x .* x, 2);
        scores = ((x2 - n).^2).^(alpha) + (0.5*x2 + sum(x,2))/n + 0.5;

    % Periodic Function
    case 16
        sin2x = sin(x) .^ 2;
        sumx2 = sum(x .^2, 2);
        scores = 1 + sum(sin2x, 2) - 0.1 * exp(-sumx2);
  
    
    % Rastrigin Function
    case 17
        A = 10;
        scores = (A * n) + (sum(x .^2 - A * cos(2 * pi * x), 2));
   
    
    % Xin-She Yang (Function 2)
    case 18
        scores = sum(abs(x), 2) .* exp(-sum(sin(x .^2), 2));
    
   
    % Schwefel 1.2 Function
    case 19
        scores = 0;
        for i = 1:n
            scores = scores + sum(x(1:i))^2;
        end
    
   
    % Step 2 Function
    case 20
        scores = sum(floor(x + 0.5) .^ 2, 2);     
    
   
    % Penalized 2 Function
    case 21
        scores = sin(3 * pi * x(1)) ^ 2;
        for i = 1:n-1
            scores = scores + ( x(i)-1 ) ^ 2 * (1 + sin(3 * pi * x(i+1)) ^2);
        end
        scores = (scores + (x(n) - 1) ^ 2 * (1 + sin(2 * pi * x(n))^2)) * 0.1;
        for i = 1:n
            if x(i) > 5
                scores = scores + 100 * (x(i) - 5) ^ 4;
            elseif x(i) < -5
                scores = scores + 100 * ( -x(i) - 5) ^ 4;
            end
        end
          
end
% switch end

% 获取测试函数�?���?
[~, ~, opt_f] = Get_Func_Info(n, FuncNo);

% 计算误差�?
fitness_error = abs(scores - opt_f);

end