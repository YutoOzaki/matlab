function userscript
    %% define data
    N = 500;
    D = 2;
    numclass = 10;
    data = testdata(N, D, numclass);
    
    %% define model
    J = 13;
    numnode = [D; 30; J; 20; D];
    
    K = 10;
    PI = rand(K, 1);
    PI = PI./sum(PI);
    
    eta_mu = zeros(J, K);
    eta_sig = zeros(J, K);
    for k=1:K
        eta_mu(:, k) = mvnrnd(zeros(1, J), diag(3.*ones(J, 1)), 1)';
        eta_sig(:, k) = mvnrnd(zeros(1, J), diag(ones(J, 1)), 1).^2';
    end
    
    L = 2;
    
    encnet = struct(...
        'connect', lineartrans(numnode(2), numnode(1)),...
        'activate', tanhtrans(),...
        'mu', lineartrans(numnode(3), numnode(2)),...
        'sig', lineartrans(numnode(3), numnode(2)),...
        'exp', exptrans(),...
        'reparam', reparamtrans(numnode(3), L)...
        );
    
    decnet = struct(...
        'connect', lineartrans(numnode(4), numnode(3)),...
        'activate', tanhtrans(),...
        'mu', lineartrans(numnode(5), numnode(4)),...
        'sig', lineartrans(numnode(5), numnode(4)),...
        'exp', exptrans()...
        );
    
    priornet = struct(...
        'reparam', reparamtrans(numnode(3), 1),...
        'weight', mogtrans(eta_mu, eta_sig, PI)...
        );
    
    lossnode = lossfunc();
    
    names = fieldnames(encnet);
    for i=1:length(names)
        encnet.(names{i}).init();
    end
    
    names = fieldnames(decnet);
    for i=1:length(names)
        decnet.(names{i}).init();
    end
    
    names = fieldnames(priornet);
    for i=1:length(names)
        priornet.(names{i}).init();
    end
    
    %% define configuration
    numepoch = 100;
    batchsize = 20;
    numbatch = floor(N / batchsize);
    batchidx = zeros(2, 1);
    
    %% main loop
    loss = zeros(numepoch, 1);
            
    for epoch=1:numepoch
        rndidx = randperm(N);
        
        for batch=1:numbatch
            batchidx(1) = batchidx(2) + 1;
            batchidx(2) = batchidx(1) + batchsize - 1;
            x = data(:, rndidx(batchidx(1):batchidx(2)));
            
            % forward propagation
            loss(epoch) = loss(epoch) + fprop(x, encnet, decnet, priornet, lossnode);
            
            % backward propagation
            bprop(x, encnet, decnet, priornet, lossnode);
            
            % gradient checking
            gradcheck(x, encnet, decnet, priornet, lossnode);
            
            % update
            
            
            pause
        end
        
        figure(1); plot(loss(1:epoch)); drawnow;
        fprintf('epoch %d, loss: %e\n', epoch, loss(epoch));
        
        batchidx = batchidx .* 0;
    end
end
