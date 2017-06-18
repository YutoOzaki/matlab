function expconj_demo
    %% Setup
    p = zeros(2, 1);

    D = 4;
    sig = iwishrnd(diag(rand(D, 1)), 7);
    mu_0 = rand(D, 1);
    m = 0.8;
    mu = mvnrnd(mu_0', sig./m)';

    %% pdf notation
    p(1) = mvnpdf(mu', mu_0', sig./m);

    %% exponential notation 1 (fixed variance and covariance)
    pmt = inv(sig);
    h = 1/sqrt(det(2*pi/m*sig));
    
    lambda_1 = m.*mu_0;
    eta = pmt*mu;
    
    lambda_2 = m;
    a_eta = 0.5*mu'*pmt*mu;
    
    a_lambda = 0.5*m*mu_0'*pmt*mu_0;
    
    p(2) = h*exp(lambda_1'*eta - lambda_2*a_eta - a_lambda);

    %% relative error
    re = abs(p(1) - p(2))/max(p(1), p(2));disp(p);
    
    fprintf('relative error among pdf calculations\n'); 
    fprintf(' pd\n'); disp(p'); 
    fprintf(' relative error\n'); disp(re');
end