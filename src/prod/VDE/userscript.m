function userscript(numepoch)
    %% load data
    load('testdata.mat');
    N = size(data, 2);
    
    %% define model
    nets = struct();
    L = 1;
    
    load('pretrained.mat');
    netnames = fieldnames(bestprms);
    for i=1:length(netnames)
        nets.(netnames{i}) = bestprms.(netnames{i});
        
        nodenames = fieldnames(nets.(netnames{i}));
        for j=1:length(nodenames)
            if ~isempty(nets.(netnames{i}).(nodenames{j}).optm)
                nets.(netnames{i}).(nodenames{j}).setoptm(adagrad(1e-2, 1e-8, 'asc'));
            end
        end
    end
    
    load('fit.mat');
    nodenames = fieldnames(bestprms);
    for i=1:length(nodenames)
        nets.(nodenames{i}) = bestprms.(nodenames{i});
        
        nodenames = fieldnames(nets.(netnames{i}));
        for j=1:length(nodenames)
            if ~isempty(nets.(netnames{i}).(nodenames{j}).optm)
                nets.(netnames{i}).(nodenames{j}).setoptm(adagrad(1e-4, 1e-8, 'asc'));
            end
        end
    end
    
    nets.encrpm.reparam.L =  L;
    nets.priornet.reparam.L = 1;
    K = nets.priornet.weight.K;
    
    lossnode = lossfun();
    
    %% define configuration
    batchsize = 100;
    numbatch = floor(N / batchsize);
    batchidx = zeros(2, 1);
    
    %% main loop
    loss = zeros(numepoch, 1);
    label = zeros(N, 1);
    z = zeros(nets.priornet.weight.J, N);
            
    for epoch=1:numepoch
        rndidx = randperm(N);
        
        if rem(floor(epoch/20), 2) == 0
                prms_tbu = struct(...
                    'encnet', nets.encnet,...
                    'encrpm', nets.encrpm,...
                    'decnet', nets.decnet,...
                    'decrpm', nets.decrpm...
                );
            else
                prms_tbu = struct(...
                    'priornet', nets.priornet...
                );
        end
            
        for batch=1:numbatch
            batchidx(1) = batchidx(2) + 1;
            batchidx(2) = batchidx(1) + batchsize - 1;
            x = data(:, rndidx(batchidx(1):batchidx(2)));
            
            % forward propagation
            nets.encrpm.reparam.init();
            nets.priornet.reparam.init();
            loss(epoch) = loss(epoch) + fprop(x, nets, lossnode);
            
            % backward propagation
            bprop(nets, lossnode);
            
            % gradient checking
            %gradcheck(x, nets, lossnode);pause
            
            % update
            update(nets);
            %update(prms_tbu);
            
            % clustering assignment
            [gam, zt] = posterior(x, nets.encnet, nets.encrpm, nets.priornet);
            [~,I] = max(gam);
            label(rndidx(batchidx(1):batchidx(2)), 1) = I';
            z(:, rndidx(batchidx(1):batchidx(2))) = zt;
        end
        
        figure(1); plot(loss(1:epoch)); drawnow;
        fprintf('epoch %d, loss: %e\n', epoch, loss(epoch));
        
        c = rand(K, 3);
        for k=1:K
            idx = label == k;
            
            figure(2);
            scatter(data(1, idx), data(2, idx), 20, c(k, :)./sum(c(k, :)));hold on
            
            figure(3);
            scatter(z(1, idx), z(2, idx), 20, c(k, :)./sum(c(k, :))); hold on
            scatter(nets.priornet.weight.prms.eta_mu(1, k), nets.priornet.weight.prms.eta_mu(2, k),...
                    20, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', [0 .7 .7], 'LineWidth', 2); hold on
            drawellip(nets.priornet.weight.prms.eta_mu(:, k), diag(exp(nets.priornet.weight.prms.eta_lnsig(:, k)))); hold on
        end
        hold off
        drawnow();
        
        batchidx = batchidx .* 0;
    end
    
    save('vde_clustering_result.mat', 'label');
end

function [gam, z] = posterior(x, encnet, encrpm, priornet)
    names = fieldnames(encnet);
    input = x;
    for i=1:length(names)
        input = encnet.(names{i}).forwardprop(input);
    end

    mu = encrpm.mu.forwardprop(input);
    sigsq = encrpm.exp.forwardprop(encrpm.lnsigsq.forwardprop(input));
    
    z = priornet.reparam.forwardprop(struct('mu', mu, 'sig', sigsq));
    z = mean(z, 3);
    gam = priornet.weight.forwardprop(z);
end