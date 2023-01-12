function [lb, ub, opt_f] = Get_Func_Info(Dim, FuncNo)

% Dim: 问题维度
% FuncNo: 函数编号

switch FuncNo
    
    % Brown Function
    case 1
        lb = -1;
        ub = 4;
        opt_f = 0;
    
    % Exponential Function
    case 2
        lb = -1;
        ub = 1;
        opt_f = 1;
    
    % Griewank Function: 【原�?
    case 3
        lb = -100;
        ub = 100;
        opt_f = 0;
    
    
    
    % Ridge Function：岭回归函数
    case 4
        lb = -5;
        ub = 5;
        opt_f = 0;
        
    % Schwefel 2.20 Function
    case 5
        lb = -100;
        ub = 100;
        opt_f = 0;
    
    % Schwefel 2.21 Function：�?原�?
    case 6
        lb = -100;
        ub = 100;
        opt_f = 0;
    
    % Schwefel 2.22 Function：�?原�?
    case 7
        lb = -100;
        ub = 100;
        opt_f = 0;
    
    % Schwefel 2.23 Function
    case 8
        lb = -10;
        ub = 10;
        opt_f = 0;
    
    % Sphere Function：�?原�?
    case 9
        lb = 0;
        ub = 10;
        opt_f = 0;
    
    % Sum Squares Function
    case 10
        lb = -10;
        ub = 10;
        opt_f = 0;
    
    % Xin-She Yang (Function 3)
    case 11
        lb = -20;
        ub = 20;
        opt_f = -1;
    
    % Zakharov Function
    case 12
        lb = -5;
        ub = 10;
        opt_f = 0;
    
    % Ackley 1 Function：�?原�?
    case 13
        lb = -35;
        ub = 35;
        opt_f = 0;
    
    
   
    % Alpine 1 Function
    case 14
        lb = -10;
        ub = 10;
        opt_f = 0;
    
   
    
    % Happy Cat  function
    case 15
        lb = -2;
        ub = 2;
        opt_f = 0;
    
    % Periodic Function：本来是 2-Dim, 改成 D-Dim
    case 16
        lb = -10;
        ub = 10;
        opt_f = 0.9;
    
   
    
    % Rastrigin Function：�?原�?
    case 17
        lb = -5.12;
        ub = 5.12;
        opt_f = 0;
    
  
    
    % Xin-She Yang (Function 2)
    case 18
        lb = -2 * pi;
        ub = 2 * pi;
        opt_f = 0;
    
   
    % Schwefel 1.2 Function：�?原�?
    case 19
        lb = -100;
        ub = 100;
        opt_f = 0; 
    
   
        
    % Step 2 Function：�?原�?
    case 20
         lb = -100;
        ub = 100;
        opt_f = 0;       
    
    
    
    % Penalized 2 Function：�?原�?
    case 21
        lb = -50;
        ub = 50;
        opt_f = 0;
        
end

end