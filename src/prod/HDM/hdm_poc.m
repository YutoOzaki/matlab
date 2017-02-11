function hdm_poc
%% setup
N = 200;
eta = 4.0;

K = 8;
M = 5;
beta = [8 6 2 4 1 3 2 2];
gamma_ = 10;
alpha_3 = 1;
alpha_2 = 100;
alpha_1 = 8;

numsticks = 1000;

pi_cul = zeros(numsticks, 1);

%% Chinese restaurant process (super-categories)
doc_sc = zeros(N,1);
n_s = [];
doc_sc(1) = 1;
n_s(1) = 1;
for n=2:N
    numsc = length(n_s);
    p = rand();
    p_k = zeros(numsc + 1,1);
    p_k(1) = n_s(1);
    for k=2:numsc
        p_k(k) = p_k(k-1) + n_s(k);
    end
    p_k(numsc + 1) = p_k(numsc) + eta;
    p_k = p_k./(n-1+eta);
    [row,~] = find(p_k > p);
    if length(row) == 1
        n_s(numsc + 1) = 1;
        doc_sc(n) = numsc + 1;
    else
        n_s(row(1)) = n_s(row(1)) + 1;
        doc_sc(n) = row(1);
    end
end
figure(1);
stem(n_s,'Marker','None');title(strcat('super-categories (',num2str(numsc),' categories in total)'));
xlim([0 numsc+1]);
ax = gca; ax.TickLength = [0 0.025];

%% Chinese restaurant process (basic-level categories)
doc_bc = zeros(N,1);
n_b = cell(numsc,1);
for n=1:N
    sc = doc_sc(n);
    
    numbc = length(n_b{sc});
    if numbc == 0
        n_b{sc}(1) = 1;
        doc_bc(n) = sc + 0.1;
    else
        p = rand();
        p_k = zeros(numbc + 1,1);
        p_k(1) = n_b{sc}(1);
        for k=2:numbc
            p_k(k) = p_k(k-1) + n_b{sc}(k);
        end
        p_k(numbc + 1) = p_k(numbc) + eta;
        p_k = p_k./(sum(n_b{sc})-1+eta);
        [row,~] = find(p_k > p);
        if length(row) == 1
            n_b{sc}(numbc + 1) = 1;
            doc_bc(n) = sc + 0.1*(numbc + 1);
        else
            n_b{sc}(row(1)) = n_b{sc}(row(1)) + 1;
            doc_bc(n) = sc + 0.1*row(1);
        end
    end
end

figure(2)
numtc = 0;
x_start = 1;
for i=1:numsc
    numbc = length(n_b{i});
    numtc = numtc + numbc;
    x_end = x_start + numbc - 1;
    stem(x_start:x_end,n_b{i},'Marker','None');hold on;
    x_start = x_start + numbc;
end
hold off;
title(strcat('basic-lebel categories (',num2str(numtc),' categories in total)'));
xlim([0 numtc]);
ax = gca; ax.TickLength = [0 0.025];

%% generate G_g (global)
pi_g = sbp(gamma_, numsticks);
G_g = dhiricletrnd(beta, numsticks);
pi_cul = pi_cul.*0;
pi_cul(1) = pi_g(1);
for i=2:numsticks
    pi_cul(i) = pi_cul(i-1) + pi_g(i);
end

%% generate G_s (super categories)
pi_s = sbp(alpha_3, numsticks);
G_s = 0.*G_g;
randprob = rand(numsticks,1);
for i=1:numsticks
    [row,~] = find(pi_cul > randprob(i));
    G_s(i,:) = G_g(row(1),:);
end
pi_cul = pi_cul.*0;
pi_cul(1) = pi_s(1);
for i=2:numsticks
    pi_cul(i) = pi_cul(i-1) + pi_s(i);
end

%% generate G_c
pi_c = sbp(alpha_2, numsticks);
G_c = 0.*G_g;
randprob = rand(numsticks,1);
for i=1:numsticks
    [row,~] = find(pi_cul > randprob(i));
    G_c(i,:) = G_s(row(1),:);
end
pi_cul = pi_cul.*0;
pi_cul(1) = pi_c(1);
for i=2:numsticks
    pi_cul(i) = pi_cul(i-1) + pi_c(i);
end

%% generate G_n
theta_nt = sbp(alpha_1, numsticks);
G_n = 0.*G_g;
randprob = rand(numsticks,1);
for i=1:numsticks
    [row,~] = find(pi_cul > randprob(i));
    G_n(i,:) = G_c(row(1),:);
end
pi_cul = pi_cul.*0;
pi_cul(1) = theta_nt(1);
for i=2:numsticks
    pi_cul(i) = pi_cul(i-1) + theta_nt(i);
end

%% generate phi
h_3 = zeros(1,K);
rndidx= randi(numsticks,numsticks,1);
for i=1:M
    phi = G_n(rndidx(i),:);
    h_3 = h_3 + mnrnd(1,phi);
end

%%{
numplt = 5;
figure(3);
subplot(numplt,1,1);stem(G_g(:,1), pi_g, 'Marker', 'None');xlim([0 1]);
ax = gca; ax.TickLength = [0 0.025];
subplot(numplt,1,2);stem(G_s(:,1), pi_s, 'Marker', 'None');xlim([0 1]);
ax = gca; ax.TickLength = [0 0.025];
subplot(numplt,1,3);stem(G_c(:,1), pi_c, 'Marker', 'None');xlim([0 1]);
ax = gca; ax.TickLength = [0 0.025];
subplot(numplt,1,4);stem(G_n(:,1), theta_nt, 'Marker', 'None');xlim([0 1]);
ax = gca; ax.TickLength = [0 0.025];
subplot(numplt,1,5);stem(h_3, 'Marker', 'None');
ax = gca; ax.TickLength = [0 0.025];
%}

%DPは離散分布を生成する: sbpでstickをN個生成 -> uniform乱数でヒットしたところから基底分布上のpi_ ~ dhiricletrndを返す

%% backward propagation (inefernce)

%% evaluaion of objective function
end

function pi_ = dhiricletrnd(alpha, N)
    alpha = alpha(:)';
    pi_ = gamrnd(repmat(alpha, N, 1), 1);
    pi_ = pi_./repmat(sum(pi_,2), 1, length(alpha));
    
    %{
    y = gamma(sum(alpha))/prod(gamma(alpha))*prod(pi_.^(alpha - 1));
    %}
end

function pi_ = sbp(alpha, N)
    pi_ = zeros(N,1);
    v = betarnd(1, alpha, [N, 1]);
    
    pi_(1) = v(1);
    v_prod = 1 - v(1);
    
    for i=2:N
        pi_(i) = v(i)*v_prod;
        v_prod = v_prod * (1 - v(i));
    end
    
    %{
    mu = 0; sig = 1;
    subplot(311); stem(1:N,pi_,'Marker','None');
    subplot(312); plot(1:N,log(pi_));
    
    y = normrnd(mu,sig,[N,1]);
    x = linspace(-4, 4, N);
    subplot(313); stem(y, pi_, 'Marker', 'None'); hold on
    plot(x, normpdf(x, mu, sig));hold off
    %}
end