function [ candidate ] = Candidate(i,D,prob,f1,x1,gbest1,mindex1)
%UNTITLED4 ดหดฆฯิสพำะนุดหบฏสตฤีชาช
%   ดหดฆฯิสพฯ๊ฯธหตร๗
[row,col]=size(x1);
beta=1.5;
z= levy(1,D,beta);
%gau=Gauissian( x2, N2,D);

a=rand();
if prob>a
    
        candidate=gbest1.*z;   %ฑรโฯศ๋พึฒฟื๎ำลฃฌดำพซำขึึศบักศกพซำขมฃืำ

else
    
    n1=randi(row);
    n2=randi(row);
    while n1==n2 || n1==mindex1 || n2==mindex1
        n1=randi(row);
        n2=randi(row);
    end
    if f1(n1)<f1(n2)
        c=x1(n1);
        
            candidate=c.*z;
            %candidate=c.*z*0.01;
       
    else
        d=x1(n2);
        
            candidate=d.*z;
            %candidate=d.*z*0.01;
    end
end

end

