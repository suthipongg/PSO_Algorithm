%%
%   func_num  ------Test function number
%   lb   -----------lower bound of the independent variable
%   rb   -----------Upper bound of independent variable
%   N    -----------initial number of particles
%   D    -----------problem dimension
%   c    -----------Population proportional coefficient
%   maxgen    ------The maximum number of iterations
%%parameters initialization
clc;
clear;
for ii=1:6
    for ij=1
        func_num=ii;
        c1=1.5;
        c2=1.5;           %acceleration factor
        w=0.5;              %inertia weight
        N=50;
        D=2;
        c=0.5;
        %maxgen=D*10^4/N;
        maxgen=50;
        [L,U,opt_f]=Get_Func_Info(D, func_num);
        lb=U(1);
        rb=L(1);
        lv=0.2*(lb-rb);       %maximum particle speed
        rv=-lv;               %minimum particle speed
        F=0.5;                %scaling factor
        x=zeros(N,D);
        fitness=zeros(N,1);
        v=zeros(N,D);
        k = [
            [0.0352792, 0.0199591],
            [0.972657, 0.13886],
            [0.146975, 0.12694],
            [0.00279465, 0.000736522],
            [0.00377171, 0.482728],
            [0.134356, 0.387609],
            [0.0144807, 0.133271],
            [0.983048, 0.904877],
            [0.00276271, 0.0142066],
            [0.305827, 0.980386],
            [0.146731, 0.146548],
            [0.144717, 0.39138],
            [0.1458, 0.0178036],
            [0.864824, 0.0523703],
            [0.146181, 0.137691],
            [0.146136, 0.145907],
            [0.145968, 0.129398],
            [0.145205, 0.160464],
            [0.145922, 0.145953],
            [0.145769, 0.130206],
            [0.130404, 0.191958],
            [0.407387, 0.819565],
            [0.00260948, 0.0709879],
            [0.302574, 0.676513],
            [0.00259493, 0.238053],
            [0.00370858, 0.487809],
            [0.412087, 0.252487],
            [0.0033577, 0.981942],
            [0.0038307, 0.00375424],
            [0.00302134, 0.75112],
            [0.00299116, 0.063875],
            [0.00354017, 0.753571],
            [0.0191197, 0.00347963],
            [0.0035579, 0.284287],
            [0.00192326, 0.00189245],
            [0.303276, 0.740176],
            [0.00180084, 0.392396],
            [0.00173977, 0.131453],
            [0.017304, 0.118931],
            [0.00161805, 0.0430139],
            [0.00155664, 0.00447005],
            [0.303643, 0.669725],
            [0.00143463, 0.417026],
            [0.00137375, 0.751343],
            [0.0169374, 0.00128227],
            [0.00125145, 0.186952],
            [0.00119041, 0.432895],
            [0.00113004, 0.284186],
            [0.00106857, 0.551651],
            [0.00100774, 0.4668],
            [0.000946417, 0.520734]];
        %%initialization
        for i=1:N
            x(i,:)=(lb-rb).*k(i, :)+rb;              %Particle position initialization
            %x(i,:)=(lb-rb).*rand(1,D)+rb;
            fitness(i)=TestFunction(x(i,:),func_num);   %Calculate fitness value
            v(i,:)=rands(1,D).*lv;                         %Speed initialization
        end
        %%Calculate particle local and global optima
        pbest=x;                                % list position all particle
        pbestfitness=fitness;                   % list particle local optimum all particle
        [gbestfitness,index]=min(pbestfitness); % Particle Global Optimum
        gbest=x(index,:);                       % Global Position
        
        N1=c*N;                 % N dominant
        N2=N-N1;                % N poor
        prevfit=gbestfitness;   % P gd
        tnum=0;                 % M
        iter=1;
       
        %%cycle
        while iter<=maxgen
            r1=rand(1,D);           % random for undate velocity poor particle
            r2=rand(1,D);           % random for undate velocity poor particle
           
            
            %%Divide populations based on competition mechanism
            [best,i1]=sort(fitness);      % sort

            %dominant population
            x1=x(i1(1:N1),:);               % separate dominant particle
            f1=fitness(i1(1:N1));           % fitness dominant
            v1=v(i1(1:N1),:);               % velocity dominant
            pbest1=pbest(i1(1:N1),:);       % pbest dominant
            ave(iter)=mean(f1);             % mean fitness dominant
            [gbestfitness1,mindex1]=min(f1);% gbest fitness dominant
            gbest1=x1(mindex1,:);           % gbest position dominant

            %poor population
            x2=x(i1(N1+1:N),:);             % separate poor particle
            pbest2=pbest(i1(N1+1:N),:);     % pbest poor
            f2=fitness(i1(N1+1:N));         % fitness poor
            v2=v(i1(N1+1:N),:);             % velocity poor
            [gbestfitness2,mindex2]=min(f2);% gbest fitness poor
            gbest2=x2(mindex2,:);           % gbest position poor

            %%%%Different populations use different evolutionary strategies to evolve separately
            %%Elastic candidate learning stategy(Population 2)

            % compute poor velocity
            for i=1:N2
                prob=0.5;
                candidate=Candidate(i,D,prob,f1,x1,gbest1,mindex1);
                v2(i,:)=w*v2(i,:)+c1*r1.*(pbest2(i,:)-x2(i,:))+c2*r2.*(candidate-x2(i,:));
           
                %Speed, position limit
                v2(i,find(v2(i,:)>lv))=lv;                         %maximum speed limit
                v2(i,find(v2(i,:)<rv))=rv;
                
                x2(i,:)=x2(i,:)+v2(i,:);
                x2(i,find(x2(i,:)>lb))=lb;                         %particle position limit
                x2(i,find(x2(i,:)<rb))=rb;
            end
            
            %%Average dimension learning (population 1)
            rowsum=sum(x1,2);  %Calculate the sum of each row
            colsum=sum(x1);    %Calculate the sum of each column

            % compute dominant velocity
            for i=1:N1
                for j=1:D
                    S(j)=1/(1+exp(-(x1(i,j)-mean(x1(i,:)))));
                    rn=rand();
                    mbest(i,j)=S(j)*rn*rowsum(i)/(D)+(1-S(j))*(1-rn)*colsum(j)/(N1) ;
                end
                tn(i)=1/(1+exp(-(f1(i)/ave(1))))^iter;
                v1(i,:)=w*v1(i,:)+c1*r1.*(mbest(i,:)-x1(i,:));    %Particle Velocity Update
                %Speed, position limit
                v1(i,find(v1(i,:)>lv))=lv;                         %maximum speed limit
                v1(i,find(v1(i,:)<rv))=rv;
                
                if tnum>6
                    m=randi(N1,1);
                    n=randi(N1,1);
                    while m==i || m==n ||n==i
                        m=randi(N1,1);
                        n=randi(N1,1);
                    end
                  x1(i,:)=gbest+F*(x1(m,:)-x1(n,:));
                  %particle position update
                else
                    x1(i,:)=tn(i)*x1(i,:)+v1(i,:);                   %particle position update
                end
                x1(i,find(x1(i,:)>lb))=lb;                           %particle position limit
                x1(i,find(x1(i,:)<rb))=rb;
            end
            
            
            x=[x1;x2];
            v=[v1;v2];
            pbest=[pbest1;pbest2];
            
            for i=1:N
                fitness(i)=TestFunction(x(i,:), func_num);
            end
            %%Update local and global optima of particles
            for j=1:N
                if fitness(j)<pbestfitness(j)
                    pbest(j,:)=x(j,:);
                    pbestfitness(j)=fitness(j);
                end
                if fitness(j)<gbestfitness
                    gbest=x(j,:);
                    gbestfitness=fitness(j);
                end
            end
            if gbestfitness<prevfit
                prevfit=gbestfitness;
                tnum=0;
            else
                tnum=tnum+1;
            end
            
            % output the result of each iteration
            [Gbest_Fitness_new,I]=min(fitness);
            shmpso_Best_fitness(iter)=Gbest_Fitness_new;%Record the best fitness value of each generation
            result(iter)=gbestfitness;
            iter=iter+1;
            
        end
   
        fprintf('problem:%d times:%d  results:%d\n',ii,ij,Gbest_Fitness_new);   
        shmpso_result(ij,ii) = min(Gbest_Fitness_new);    
     
    end
 
end

