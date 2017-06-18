function expmvn_demo
    %% Setup
    p = zeros(3, 1);
    re = zeros(3, 1);

    D = 4;
    sig = iwishrnd(diag(rand(D, 1)), 7);
    mu_0 = rand(D, 1);
    m = 0.8;
    mu = mvnrnd(mu_0', sig./m)';
    x = mvnrnd(mu', sig)';

    %% pdf notation
    p(1) = mvnpdf(x', mu', sig);

    %% exponential notation 1 (fixed variance and covariance)
    pmt = inv(sig);
    h = 1/sqrt(det(2*pi*sig)) * exp(-0.5*x'*pmt*x);
    eta = pmt*mu;
    t = x;
    a = 0.5*mu'*pmt*mu;
    p(2) = h*exp(eta'*t - a);

    %% exponential notation 2 (fixed variance and covariance)
    pmt = inv(sig);
    h = 1/sqrt(det(2*pi*sig)) * exp(-0.5*x'*pmt*x);
    eta = pmt*mu;
    t = x;
    a = 0.5*eta'*sig*eta;
    p(3) = h*exp(eta'*t - a);

    %% relative error
    re(1) = abs(p(1) - p(2))/max(p(1), p(2));
    re(2) = abs(p(2) - p(3))/max(p(2), p(3));
    re(3) = abs(p(1) - p(3))/max(p(1), p(3));
    
    fprintf('relative error among pdf calculations\n'); 
    fprintf(' pd\n'); disp(p'); 
    fprintf(' relative error\n'); disp(re');
end