function VAE_demo
    %% load data
    datafilepath = userpath;
    datafilepath = datafilepath(1:end - 1);
    datafilepath = strcat(datafilepath, '\data\MNIST\MNISTdataset.mat');
    
    load(datafilepath);
    
    %% training
    %traindata = traindata(:,1:20000); % test with small dataset
    training(traindata);
end

function training(traindata)
    rng(1);
     %% setup
    N = size(traindata, 2);
    D = size(traindata, 1);
    
    maxepoch = 100;
    batchsize = 100;
    assert(0 == rem(N, batchsize), 'N/batchsize should be integer');
    numbatch = N/batchsize;
    H = 500;
    J = 2;
    L = 1;
    
    encoder = 'gaussian';
    decoder = 'gaussian';
    %decoder = 'bernoulli';
    
    % data transformation
    if strcmp(decoder, 'bernoulli')
        flooridx = traindata < 0.5;
        ceilidx = traindata >= 0.5;
        traindata(flooridx) = 0;
        traindata(ceilidx) = 1;
    end
    
    % parameters
    if strcmp(encoder, 'gaussian')
        encode = @gaussiantrans;
        phi = struct(...
            'W_in', initprm(H, D), 'b_in', initprm(H, 1),...
            'W_mu', initprm(J, H), 'b_mu', initprm(J, 1),...
            'W_sg', initprm(J, H), 'b_sg', initprm(J, 1));
    elseif strcmp(encoder, 'bernoulli')
    end
    dphi = phi;
    phinames = fieldnames(phi);
    
    if strcmp(decoder, 'gaussian')
        decode = @gaussiantrans;
        evalelbo = @gaussianelbo;
        decoderdelta = @gaussiandelta;
        sampler = @gausssampler;
        
        theta = struct(...
            'W_in', initprm(H, J), 'b_in', initprm(H, 1),...
            'W_mu', initprm(D, H), 'b_mu', initprm(D, 1),...
            'W_sg', initprm(D, H), 'b_sg', initprm(D, 1));
    elseif strcmp(decoder, 'bernoulli')
        decode = @bernoullitrans;
        evalelbo = @bernoullielbo;
        decoderdelta = @bernoullidelta;
        sampler = @bernoullisampler;
        
        theta = struct(...
            'W_in', initprm(H, J), 'b_in', initprm(H, 1),...
            'W_p', initprm(D, H), 'b_p', initprm(D, 1));
    end
    dtheta = theta;
    thetanames = fieldnames(theta);
    
    % optimizer
    update = @adam;
    modelenc.eps = 1e-7;
    modelenc.eta = -1e-3;
    modelenc.beta_1 = 0.9;
    modelenc.beta_2 = 0.999;
    modelenc.m = dphi;
    modelenc.v = dphi;
    modelenc.t = 1;
    
    modeldec = modelenc;
    modeldec.m = dtheta;
    modeldec.v = dtheta;
    
    for i=1:numel(phinames)
        modelenc.m.(phinames{i}) = modelenc.m.(phinames{i}) .* 0;
        modelenc.v.(phinames{i}) = modelenc.v.(phinames{i}) .* 0;
    end
    
    for i=1:numel(thetanames)
        modeldec.m.(thetanames{i}) = modeldec.m.(thetanames{i}) .* 0;
        modeldec.v.(thetanames{i}) = modeldec.v.(thetanames{i}) .* 0;
    end
    
    %{
    update = @rmsprop;
    modelenc.gamma = 0.9;
    modelenc.eta = -1e-3;
    modelenc.eps = 1e-6;
    modelenc.Eg2t = dphi;
    names = fieldnames(modelenc.Eg2t);
    for i=1:numel(names)
        modelenc.Eg2t.(names{i}) = modelenc.Eg2t.(names{i}) .* 0;
    end
    
    modeldec.gamma = 0.9;
    modeldec.eta = -1e-3;
    modeldec.eps = 1e-6;
    modeldec.Eg2t = dtheta;
    names = fieldnames(modeldec.Eg2t);
    for i=1:numel(names)
        modeldec.Eg2t.(names{i}) = modeldec.Eg2t.(names{i}) .* 0;
    end
    %}
    
    %% main
    mu = zeros(J, batchsize);
    cov = ones(J, batchsize);
    e = zeros(J, batchsize, L);
    losshist = zeros(maxepoch, 1);
    
    for epoch = 1:maxepoch
        rndidx = randperm(N);
        batchidx = 1:batchsize;
        totalloss = 0;
        
        tic;
        for i=1:numbatch
            % initialize delta
            for j=1:numel(thetanames); dtheta.(thetanames{j}) = 0; end
            for j=1:numel(phinames); dphi.(phinames{j}) = 0; end
            
            % get input
            x = traindata(:, rndidx(batchidx));
            
            % sample noise for reparameterization trick
            e = noisernd(e, mu, cov);
            
            % get parameters for prior distribution
            enc = encode(x, phi);
            
            % evaluate Kullback-Leibler Divergence (first term of loss function)
            kld = -mvnkld(enc.mu, enc.sg.^2, mu, cov);
            
            % backpropagation
            dmDds = 1./enc.sg - enc.sg;
            dmDdm = -enc.mu;
            
            elbo = 0;
            for l=1:L
                % sample z (latent vairable)
                z = enc.mu + enc.sg.*e(:,:,l);
                
                % get paramters for distribution
                dec = decode(z, theta);
                
                % evaluate variational lower boud (evidence lower bound)
                elbo = elbo + evalelbo(x, dec);
                
                % backpropagation
                decd = decoderdelta(x, z, theta, dec);
                
                for j=1:numel(thetanames); 
                    dtheta.(thetanames{j}) = dtheta.(thetanames{j}) + decd.(thetanames{j}); 
                end
                
                delta = theta.W_in' * decd.delta;
                dzdm = delta;
                dzds = delta .* e(:,:,l);
                
                dmu = dzdm + dmDdm;
                dphi.W_mu = dphi.W_mu + dmu * enc.h';
                dphi.b_mu = dphi.b_mu + dmu;
                
                dsg = 0.5 .* enc.sg .* (dzds + dmDds);
                dphi.W_sg = dphi.W_sg + dsg * enc.h';
                dphi.b_sg = dphi.b_sg + dsg;
                
                dh = phi.W_mu' * dmu + phi.W_sg' * dsg;
                dphi.W_in = dphi.W_in + (dh .* enc.delta) * x';
                dphi.b_in = dphi.b_in + dh .* enc.delta;
            end
            elbo = elbo ./ L;
            
            numsample = L * batchsize;
            for j=1:numel(thetanames); 
                dtheta.(thetanames{j}) = dtheta.(thetanames{j}) ./ numsample;
            end
            dphi.W_mu = dphi.W_mu ./ numsample;
            dphi.W_sg = dphi.W_sg ./ numsample;
            dphi.W_in = dphi.W_in ./ numsample;
            dphi.b_mu = sum(dphi.b_mu, 2) ./ numsample;
            dphi.b_sg = sum(dphi.b_sg, 2) ./ numsample;
            dphi.b_in = sum(dphi.b_in, 2) ./ numsample;
            
            loss = sum(kld + elbo)/batchsize;
            totalloss = totalloss + loss;
            
            % gradient checking
            %gradcheck(x, e, phi, theta, mu, cov, dphi, dtheta, encode, decode, evalelbo);
            
            % update
            [theta, modeldec] = update(theta, dtheta, modeldec);
            [phi, modelenc] = update(phi, dphi, modelenc);
            batchidx = batchidx + batchsize;
        end
        t = toc;
        losshist(epoch) = totalloss;
        
        fprintf('epoch %d: total loss %e (elapsed %3.3f sec)\n', epoch, totalloss, t);
        figure(1); plot(losshist); xlim([1 maxepoch]); title(sprintf('epoch %d', epoch));
        
        % draw result
        figure(2); drawsample(mu, cov, theta, decode, sampler);
        figure(3); drawrecon(x, phi, theta, e, encode, decode, sampler);
        
        if J == 2
            figure(4); drawmanifold(theta, decode, sampler);
        end
    end
end

function e = noisernd(e, mu, cov)
    mu_tmp = mu';
    cov_tmp = reshape(cov, [1 size(cov)]);
    L = size(e, 3);
    
    for l=1:L
        e(:,:,l) = mvnrnd(mu_tmp, cov_tmp)';
    end
end

function drawsample(mu, cov, theta, decode, sampler)
    cases = 25;
    
    z = mvnrnd(mu(:, 1)', diag(cov(:, 1)), cases)';
    dec = decode(z, theta);
    
    drawutil(dec, sampler)
end

function drawrecon(x, phi, theta, e, encode, decode, sampler)
    enc = encode(x, phi);
    
    z = enc.mu + enc.sg.*e(:, :, 1);
                
    dec = decode(z, theta);
    
    drawutil(dec, sampler);
end

function drawmanifold(theta, decode, sampler)
    x = linspace(-3, 3, 20);
    y = linspace(-3, 3, 20);
    
    r = length(x); c = length(y);
    
    z = zeros(2, 20);
    z(1, :) = x;
    
    canvas = zeros(28*r, 28*c);
    
    c_s = 1; c_e = 1 + 28 - 1;
    for i=1:c
        z(2, :) = z(2, :) + y(i); 
        dec = decode(z, theta);
        
        r_s = 1; 
        r_e = r_s + 28 - 1;
        for j=1:r
            canvas(r_s:r_e, c_s:c_e) = reshape(sampler(dec, j), [28 28])';
            r_s = r_s + 28;
            r_e = r_e + 28;
        end
        
        z(2, :) = z(2, :) .* 0;
        c_s = c_s + 28;
        c_e = c_e + 28;
    end
    
    imagesc(canvas);
    set(gca, 'XTick', []); set(gca, 'YTick', []); colormap gray;
    drawnow;
end

function drawutil(dec, sampler)
    cases = size(dec.h, 2);
    if cases > 25;
        cases = 25;
    end
    
    r = floor(sqrt(cases));
    c = r;
    cases = r^2;
    
    for i=1:cases
        x = sampler(dec, i);
        subplot(r, c, i); imagesc(reshape(x, [28 28])');
        set(gca, 'XTick', []); set(gca, 'YTick', []); colormap gray;
    end
    drawnow;
end

function x = gausssampler(dec, i)
    x = mvnrnd(dec.mu(:, i)', (dec.sg(:, i).^2)');
end

function x = bernoullisampler(dec, i)
    %x = binornd(1, dec.p(:, i));
    x = dec.p(:, i);
end

function [prm, model] = vanilla(prm, delta, model)
    names = fieldnames(prm);
    eta = model.eta;
    
    for i=1:numel(names)
        prm.(names{i}) = prm.(names{i}) + eta.*delta.(names{i});
    end
end

function [prm, model] = rmsprop(prm, delta, model)
    gam = model.gamma;
    alp = 1 - gam;
    eta = model.eta;
    eps = model.eps;
    Eg2t = model.Eg2t;
    
    names = fieldnames(prm);
    
    for i=1:numel(names)
        Eg2t.(names{i}) = gam.*Eg2t.(names{i}) + alp.*(delta.(names{i}).^2);
        prm.(names{i}) = prm.(names{i}) - eta .* delta.(names{i})./sqrt(Eg2t.(names{i}) + eps);
    end
    
    model.Eg2t = Eg2t;
end

function [prm, model] = adam(prm, delta, model)
    names = fieldnames(prm);
    
    beta_1 = model.beta_1;
    beta_2 = model.beta_2;
    m = model.m;
    v = model.v;
    eta = model.eta;
    eps = model.eps;
    t = model.t;
    
    for i=1:numel(names)
        m.(names{i}) = beta_1.*m.(names{i}) + (1 - beta_1).*delta.(names{i});
        v.(names{i}) = beta_2.*v.(names{i}) + (1 - beta_2).*delta.(names{i}).^2;
        
        m_tmp = m.(names{i})./(1 - beta_1^t);
        v_tmp = v.(names{i})./(1 - beta_2^t);
        
        prm.(names{i}) = prm.(names{i}) - eta .* m_tmp./(sqrt(v_tmp) + eps);
    end
    
    model.t = t + 1;
    model.m = m;
    model.v = v;
end

function W = initprm(out, in)
    W = (rand(out, in) - 0.5) .* sqrt(6 / (in + out));
end

function gradcheck(x, e, phi, theta, mu, cov, dphi, dtheta, encode, decode, evalelbo)
    fprintf('-\n');
    names = fieldnames(theta);
    fprintf(' gradient checking of theta\n');
    for i=1:numel(names)
        k = find(dtheta.(names{i})(:));
        if isempty(k)
            r = randi(size(dtheta.(names{i}), 1));
            c = randi(size(dtheta.(names{i}), 2));
        else
            L = length(k);
            I = randi(L);
            [r, c] = ind2sub(size(dtheta.(names{i})), k(I));
        end
        
        val = theta.(names{i})(r, c);
        h = 1e-4;
        
        theta.(names{i})(r, c) = val + h;
        l1 = forwardprop(x, e, phi, theta, mu, cov, encode, decode, evalelbo);
        
        theta.(names{i})(r, c) = val - h;
        l2 = forwardprop(x, e, phi, theta, mu, cov, encode, decode, evalelbo);
        dfdx = (l1 - l2)/(2*h);
        
        theta.(names{i})(r, c) = val;
        delta = dtheta.(names{i})(r, c);
        re = abs(dfdx - delta)/max(abs(dfdx), abs(delta));
        
        fprintf('  %s: %e (%5.8f, %5.8f)\n',...
            names{i}, re, dfdx, delta);
    end
    
    names = fieldnames(phi);
    fprintf(' gradient checking of phi\n');
    for i=1:numel(names)
        k = find(dphi.(names{i})(:));
        if isempty(k)
            r = randi(size(dphi.(names{i}), 1));
            c = randi(size(dphi.(names{i}), 2));
        else
            L = length(k);
            I = randi(L);
            [r, c] = ind2sub(size(dphi.(names{i})), k(I));
        end
        
        val = phi.(names{i})(r, c);
        h = 1e-4;
        
        phi.(names{i})(r, c) = val + h;
        l1 = forwardprop(x, e, phi, theta, mu, cov, encode, decode, evalelbo);
        
        phi.(names{i})(r, c) = val - h;
        l2 = forwardprop(x, e, phi, theta, mu, cov, encode, decode, evalelbo);
        dfdx = (l1 - l2)/(2*h);
        
        phi.(names{i})(r, c) = val;
        delta = dphi.(names{i})(r, c);
        re = abs(dfdx - delta)/max(abs(dfdx), abs(delta));
        
        fprintf('  %s: %e (%5.8f, %5.8f)\n',...
            names{i}, re, dfdx, delta);
    end
end

function loss = forwardprop(x, e, phi, theta, mu, cov, encode, decode, evalelbo)
    batchsize = size(x, 2);
    L = size(e, 3);
    
    enc = encode(x, phi);
    kld = -mvnkld(enc.mu, enc.sg.^2, mu, cov);
    
    elbo = 0;
    for l=1:L
        z = enc.mu + enc.sg.*e(:,:,l);
                
        dec = decode(z, theta);

        elbo = elbo + evalelbo(x, dec);
    end
    elbo = elbo./L;
    
    loss = sum(kld + elbo)/batchsize;
end

function latprm = gaussiantrans(x, prm)
    % setup
    batchsize = size(x, 2);
    
    W_in = prm.W_in;
    b_in = repmat(prm.b_in, 1, batchsize);
    W_mu = prm.W_mu;
    b_mu = repmat(prm.b_mu, 1, batchsize);
    W_sg = prm.W_sg;
    b_sg = repmat(prm.b_sg, 1, batchsize);
    
    % transformation
    a = W_in * x + b_in;
    h = tanh(a);
    
    % parameters for latent variables
    mu = W_mu * h + b_mu;
    
    logsg2 = W_sg * h + b_sg;
    sg = exp(0.5 .* logsg2);
    
    % structuring
    dhda = 1 - h.^2;
    latprm = struct('mu', mu, 'sg', sg, 'h', h, 'delta', dhda);
end

function delta = gaussiandelta(x, z, prm, dec)
    xminmu = x - dec.mu;
    sg2 = dec.sg.^2;

    dlNdmu = xminmu ./ sg2;
    dW_mu = dlNdmu * dec.h';
    db_mu = dlNdmu;
    
    dlNdsg = -0.5 .* (1 - xminmu.^2 ./ sg2);
    dW_sg = dlNdsg * dec.h';
    db_sg = dlNdsg;
    
    dlNdh = prm.W_mu' * dlNdmu + prm.W_sg' * dlNdsg;
    dW_in = (dlNdh .* dec.delta) * z';
    db_in = dlNdh .* dec.delta;

    delta = struct(...
        'W_mu', dW_mu,...
        'b_mu', sum(db_mu, 2),...
        'W_sg', dW_sg,...
        'b_sg', sum(db_sg, 2),...
        'W_in', dW_in,...
        'b_in', sum(db_in, 2),...
        'delta', dlNdh.* dec.delta);
end

function elbo = gaussianelbo(x, dec)
    D = size(x, 1);
    
    xminmu = x - dec.mu;
    sg2 = dec.sg.^2;
    elbo = -(D/2)*log(2*pi) - 0.5*sum(log(sg2)) ...
        + (-0.5 * sum(xminmu.^2 ./ sg2));
end

function latprm = bernoullitrans(x, prm)
    % setup
    batchsize = size(x, 2);
    
    W_in = prm.W_in;
    b_in = repmat(prm.b_in, 1, batchsize);
    W_p = prm.W_p;
    b_p = repmat(prm.b_p, 1, batchsize);
    
    % transformation
    a = W_in * x + b_in;
    h = tanh(a);
    
    % parameters for latent variables
    t = W_p * h + b_p;
    p = 1./(1 + exp(-t));
    
    % structuring
    dhda = 1 - h.^2;
    latprm = struct('p', p, 't', t, 'h', h, 'delta', dhda);
end

function delta = bernoullidelta(x, z, prm, dec)
    dlpdp = x./dec.p - (1 - x)./(1 - dec.p);
    dlpdt = dlpdp .* dec.p .* (1 - dec.p);
    
    dW_p = dlpdt * dec.h';
    db_p = dlpdt;
    
    dlpdh = prm.W_p' * dlpdt;
    dlpda = dlpdh .* dec.delta;
    
    dW_in = dlpda * z';
    db_in = dlpda;

    delta = struct(...
        'W_p', dW_p,...
        'b_p', sum(db_p, 2),...
        'W_in', dW_in,...
        'b_in', sum(db_in, 2),...
        'delta', dlpda);
end

function elbo = bernoullielbo(x, dec)
    elbo = sum(x.*log(dec.p) + (1 - x).*log(1 - dec.p));
end

function d = mvnkld(mu_1, cov_1, mu_2, cov_2)
    %{
    d = -0.5.*(k + sum(log(cov_1) - mu_1.^2 - cov_1));
    %}
    
    k = size(mu_1, 1);
    
    % diagonal covariance matrix version
    A = sum(cov_1./cov_2);
    B = sum((mu_2 - mu_1).^2 ./ cov_2);
    C = sum(log(cov_2) - log(cov_1));
    d = 0.5.*(A + B - k + C);
    
    % general version
    %d = 0.5.*(trace(cov_2\cov_1) + (mu_2 - mu_1)'*(cov_2\(mu_2 - mu_1)) - k + log(det(cov_2)/det(cov_1)));
end