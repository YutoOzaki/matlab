function expmat_demo
    %% Setup
    D = 4;
    mu_0 = rand(D, 1);
    sig = iwishrnd(diag(rand(D, 1)), 5);
    m = 0.2;
    pmt = inv(sig./m);
    N = 10000;
    
    mu = mvnrnd(mu_0', sig./m, N);
    
    %% E[eta]
    E_1 = sum(pmt*mu', 2)./N;
    E_2 = pmt*mu_0;
    
    fprintf('E[eta] (sampling vs. closed-form)\n');
    disp([E_1 E_2]);
    
    %% E[log-partition]
    E_1 = 0;
    tmp = mu*pmt;
    for i=1:N
        E_1 = E_1 + tmp(i, :) * mu(i, :)';
    end
    E_1 = (0.5 * E_1)/N;
    %E_2 = 0.5*(trace(pmt*sig)/m + mu_0'*pmt*mu_0);
    E_2 = 0.5*(D + mu_0'*pmt*mu_0);
    
    fprintf('E[log-partition] (sampling vs. closed-form)\n');
    disp([E_1 E_2]);
end