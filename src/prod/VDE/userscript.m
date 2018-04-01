function userscript
    %% define data
    N = 60;
    D = 3;
    numclass = 4;
    data = testdata(N, D, numclass);
    
    %% define model
    J = 2;
    numnode = [D; 15; J; 11; D];
    
    K = 4;
    p = rand(K, 1);
    
    eta_mu = zeros(J, K);
    eta_sig = zeros(J, K);
    for k=1:K
        eta_mu(:, k) = mvnrnd(zeros(1, J), diag(3.*ones(J, 1)), 1)';
        eta_sig(:, k) = mvnrnd(zeros(1, J), diag(ones(J, 1)), 1)';
    end
    
    L = 2;
    
    encnet = struct(...
        'connect', lineartrans(numnode(2), numnode(1), rmsprop(0.9, 1e-2, 1e-8)),...
        'activate', tanhtrans(),...
        'mu', lineartrans(numnode(3), numnode(2), rmsprop(0.9, 1e-2, 1e-8)),...
        'sig', lineartrans(numnode(3), numnode(2), rmsprop(0.9, 1e-2, 1e-8)),...
        'exp', exptrans(),...
        'reparam', reparamtrans(numnode(3), L)...
        );
    
    decnet = struct(...
        'connect', lineartrans(numnode(4), numnode(3), rmsprop(0.9, 1e-2, 1e-8)),...
        'activate', tanhtrans(),...
        'mu', lineartrans(numnode(5), numnode(4), rmsprop(0.9, 1e-2, 1e-8)),...
        'sig', lineartrans(numnode(5), numnode(4), rmsprop(0.9, 1e-2, 1e-8)),...
        'exp', exptrans()...
        );
    
    priornet = struct(...
        'reparam', reparamtrans(numnode(3), 1),...
        'weight', mogtrans(eta_mu, eta_sig, p, rmsprop(0.9, 1e-2, 1e-8))...
        );
    
    lossnode = lossfun();
    
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
    batchsize = 7;
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
            encnet.reparam.init();
            priornet.reparam.init();
            loss(epoch) = loss(epoch) + fprop(x, encnet, decnet, priornet, lossnode);
            
            %%%
            %{
            nets = struct('enc', encnet, 'dec', decnet, 'prior', priornet);  
            netnames = fieldnames(nets);
            for i=1:length(netnames)
                nodenames = fieldnames(nets.(netnames{i}));
                
                for j=1:length(nodenames)
                    fprintf('%s.%s.input\n', netnames{i}, nodenames{j});
                    disp(nets.(netnames{i}).(nodenames{j}).input);
                end
            end
            %}
            %%%
            
            % backward propagation
            bprop(encnet, decnet, priornet, lossnode);
            
            % gradient checking
            %gradcheck(x, encnet, decnet, priornet, lossnode);
            
            % update
            update(struct('encnet', encnet, 'decnet', decnet, 'priornet', priornet));
        end
        
        figure(1); plot(loss(1:epoch)); drawnow;
        fprintf('epoch %d, loss: %e\n', epoch, loss(epoch));
        
        batchidx = batchidx .* 0;
    end
end