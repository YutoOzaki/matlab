function u = fpropLpUnit(input,prms,T)
    batchSize = size(input,2);
    W = prms{1};
    c = prms{2};
    
    rho = prms{3};
    p = 1 + log(1+exp(rho));
    
    cMat = repmat(c,1,batchSize);
    
    J = length(rho);
    N = size(input,1);
    u = zeros(J,batchSize,T);
    
    for t=1:T
        x = input(:,:,t);
       
        v = abs(W*x - cMat);
        for j=1:J
            u(j,:,t) = (sum(v.^p(j))/N).^(1/p(j));
        end
    end
end

%{
%% Unit testing for gradient checking about derivatives of p-norm

N = randi(20);
p = rand + randi(4);
x = rand(1,N);

n = norm(x,p);
dn = x.*(abs(x).^(p-2))./(norm(x,p)^(p-1));

gCheck = zeros(1,N);
for i=1:N
    theta = x;
    epsilon = sqrt(eps)*abs(theta(i));    

    theta(i) = theta(i) + epsilon;
    dn1 = norm(theta,p);

    theta(i) = theta(i) - 2*epsilon;
    dn2 = norm(theta,p);

    gCheck(i) = (dn1 - dn2)/(2*epsilon);
end

rE = abs(abs(dn) - abs(gCheck))./max([abs(dn);abs(gCheck)]);

fprintf('--gradient checking--\n');
fprintf(' (N = %d, p = %3.3f)\n',N,p);
for i=1:N
    fprintf(' %d. %3.6f %3.6f %3.6e\n',i,dn(i),gCheck(i),rE(i));
end
%}

%{
%% Unit testing for gradient checking about derivatives of p-norm w.r.t. c

N = randi(20);
J = randi(10);
pvec = rand(J,1) + randi(4,J,1);
p = rand + randi(4);
a = rand(N,1);
c = rand(N,1);

n = zeros(J,1);
dn = zeros(N,J);
x = a - c;
dtheta = -1;
for i=1:J
    p = pvec(i);
    n(i) = norm(x,p);
    dn(:,i) = x.*(abs(x).^(p-2))./(norm(x,p)^(p-1));

    dn(:,i) = dn(:,i).*dtheta;
end

gCheck = zeros(N,J);
for k=1:J
    p = pvec(k);

    for i=1:N
        theta = c;
        epsilon = sqrt(eps)*abs(theta(i));

        theta(i) = theta(i) + epsilon;
        arg = a - theta;
        dn1 = norm(arg,p);

        theta(i) = theta(i) - 2*epsilon;
        arg = a - theta;
        dn2 = norm(arg,p);

        gCheck(i,k) = (dn1 - dn2)/(2*epsilon);
    end
end

rE = zeros(N,J);
for k=1:J
    rE(:,k) = abs(abs(dn(:,k)) - abs(gCheck(:,k)))./max([abs(dn(:,k));abs(gCheck(:,k))]);
end

fprintf('--gradient checking--\n');
fprintf(' N = %d\n',N);
fprintf(' p = \n');
disp(pvec');
for k=1:J
    for i=1:N
        fprintf(' (%d,%d) %3.6f %3.6f %3.6e\n',i,k,dn(i,k),gCheck(i,k),rE(i,k));
    end
end
%}

%{
%% Unit testing for gradient checking about derivatives of p-norm w.r.t. c

N = randi(20);
J = randi(10);
pvec = rand(J,1) + randi(4,J,1);
p = rand + randi(4);
a = rand(N,1);
c = rand(N,1);

n = zeros(J,1);
dn = zeros(N,J);
x = a - c;
dtheta = -1;
for i=1:J
    p = pvec(i);
    n(i) = norm(x,p);
    dn(:,i) = x.*(abs(x).^(p-2))./(norm(x,p)^(p-1));

    dn(:,i) = dn(:,i).*dtheta;
end

gCheck = zeros(N,J);
for k=1:J
    p = pvec(k);

    for i=1:N
        theta = c;
        epsilon = sqrt(eps)*abs(theta(i));

        theta(i) = theta(i) + epsilon;
        arg = a - theta;
        dn1 = norm(arg,p);

        theta(i) = theta(i) - 2*epsilon;
        arg = a - theta;
        dn2 = norm(arg,p);

        gCheck(i,k) = (dn1 - dn2)/(2*epsilon);
    end
end

rE = zeros(N,J);
for k=1:J
    rE(:,k) = abs(abs(dn(:,k)) - abs(gCheck(:,k)))./max([abs(dn(:,k));abs(gCheck(:,k))]);
end

fprintf('--gradient checking--\n');
fprintf(' N = %d\n',N);
fprintf(' p = \n');
disp(pvec');
for k=1:J
    for i=1:N
        fprintf(' (%d,%d) %3.6f %3.6f %3.6e\n',i,k,dn(i,k),gCheck(i,k),rE(i,k));
    end
end
%}

%{
%% Unit testing for gradient checking about derivatives of p-norm w.r.t. c
pnorm = @(x,p,N) (sum(abs(x).^p)/N)^(1/p);

N = randi(20);
J = randi(10);
pvec = rand(J,1) + randi(4,J,1);
p = rand + randi(4);
a = rand(N,1);
c = rand(N,1);

n = zeros(J,1);
dn = zeros(N,J);
x = a - c;
dtheta = -1;
for i=1:J
    p = pvec(i);
    n(i) = pnorm(x,p,N);
    dn(:,i) = x.*(abs(x).^(p-2))./(norm(x,p)^(p-1))./(N^(1/p));

    dn(:,i) = dn(:,i).*dtheta;
end

gCheck = zeros(N,J);
for k=1:J
    p = pvec(k);

    for i=1:N
        theta = c;
        epsilon = sqrt(eps)*abs(theta(i));

        theta(i) = theta(i) + epsilon;
        arg = a - theta;
        dn1 = pnorm(arg,p,N);

        theta(i) = theta(i) - 2*epsilon;
        arg = a - theta;
        dn2 = pnorm(arg,p,N);

        gCheck(i,k) = (dn1 - dn2)/(2*epsilon);
    end
end

rE = zeros(N,J);
for k=1:J
    rE(:,k) = abs(abs(dn(:,k)) - abs(gCheck(:,k)))./max([abs(dn(:,k));abs(gCheck(:,k))]);
end

fprintf('--gradient checking--\n');
fprintf(' N = %d\n',N);
fprintf(' p = \n');
disp(pvec');
for k=1:J
    for i=1:N
        fprintf(' (%d,%d) %3.6f %3.6f %3.6e\n',i,k,dn(i,k),gCheck(i,k),rE(i,k));
    end
end
%}

%{
%% Unit testing for gradient checking about derivatives of p-norm w.r.t. p
pnorm = @(x,p,N) (sum(abs(x).^p)/N)^(1/p);

N = randi(20);
J = randi(10);
pvec = rand(J,1) + randi(4,J,1);
p = rand + randi(4);
a = rand(N,1);
c = rand(N,1);

n = zeros(J,1);
dn = zeros(1,J);
x = a - c;

for i=1:J
    p = pvec(i);
    n(i) = pnorm(x,p,N);
    
    bufnorm = norm(x,p);
    
    nume1 = sum((abs(x).^p).*log(abs(x)));
    deno1 = bufnorm^p;
    nume2 = log(deno1);
    term1 = bufnorm*(nume1/deno1 - nume2/p);

    term2 = bufnorm * log(N) / p;
    
    const = 1/(p*(N^(1/p)));

    dn(i) = const*(term1 + term2);
end

gCheck = zeros(1,J);
arg = a - c;
for k=1:J
    p = pvec(k);

    theta = p;
    epsilon = sqrt(eps)*abs(theta);        

    theta = theta + epsilon;
    dn1 = pnorm(arg,theta,N);

    theta = theta - 2*epsilon;
    dn2 = pnorm(arg,theta,N);

    gCheck(k) = (dn1 - dn2)/(2*epsilon);
end

rE = zeros(1,J);
for k=1:J
    rE(:,k) = abs(abs(dn(:,k)) - abs(gCheck(:,k)))./max([abs(dn(:,k));abs(gCheck(:,k))]);
end

fprintf('--gradient checking--\n');
fprintf(' N = %d\n',N);
fprintf(' p = \n');
disp(pvec');

row = size(dn,1);
col = size(dn,2);

for k=1:col
    for i=1:row
        fprintf(' (%d,%d) %3.6f %3.6f %3.6e\n',i,k,dn(i,k),gCheck(i,k),rE(i,k));
    end
end
%}