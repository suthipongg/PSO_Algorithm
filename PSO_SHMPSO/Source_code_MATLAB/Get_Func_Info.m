function [lb, ub, opt_f] = Get_Func_Info(Dim, FuncNo)

% Dim: ้—ฎ้ข็ปดๅบฆ
% FuncNo: ๅฝๆ•ฐ็ผ–ๅท

switch FuncNo

    case 1
        lb = -5;
        ub = 5;
        opt_f = 0;
    

    case 2
        lb = -10;
        ub = 10;
        opt_f = 0;

    case 3
        lb = -512;
        ub = 512;
        opt_f = 0;
    

    case 4
        lb = -10;
        ub = 10;
        opt_f = 0;
        

    case 5
        lb = -5.12;
        ub = 5.12;
        opt_f = 0;
    
    case 6
        lb = -500;
        ub = 500;
        opt_f = 0;

        
end

end