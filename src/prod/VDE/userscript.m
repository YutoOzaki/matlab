function userscript(datafile, L, numepoch, gpumode, disprecon_helper, dispcluster, displabeling, evalresult)
    if ~exist('disprecon_helper', 'var')
        disprecon_helper = @defaultplot;
    end
    if ~exist('dispcluster', 'var')
        dispcluster = @dispcluster3D;
    end
    if ~exist('displabeling', 'var')
        displabeling = @displabeling3D;
    end
    if ~exist('evalresult', 'var')
        evalresult = @() fprintf('');
    end
    
    %% load data
    load(datafile);
    N = size(data, 2);
    
    if gpumode
        data = gpuArray(cast(data, 'single'));
    end
    
    %% define configuration
    batchsize = 128;
    numbatch = floor(N / batchsize);
    batchidx = zeros(2, 1);
    
    %% define model
    nets = struct();
    
    load(strcat(datafile, '_pretrained.mat'));
    netnames = fieldnames(bestprms);
    for i=1:length(netnames)
        nets.(netnames{i}) = bestprms.(netnames{i});
        
        nodenames = fieldnames(nets.(netnames{i}));
        for j=1:length(nodenames)
            if ~isempty(nets.(netnames{i}).(nodenames{j}).optm)
                nets.(netnames{i}).(nodenames{j}).setoptm(rmsprop(0.9, 1e-4, 1e-8, 'desc'));
            end
        end
    end
    
    load(strcat(datafile, '_fit.mat'));
    netnames = fieldnames(bestprms);
    for i=1:length(netnames)
        nets.(netnames{i}) = bestprms.(netnames{i});
        
        nodenames = fieldnames(nets.(netnames{i}));
        for j=1:length(nodenames)
            if ~isempty(nets.(netnames{i}).(nodenames{j}).optm)
                nets.(netnames{i}).(nodenames{j}).setoptm(rmsprop(0.9, 1e-4, 1e-8, 'desc'));
            end
        end
    end
    
    nets.encrpm.reparam.L = L;
    nets.decrpm.reparam.L = 1;
    nets.priornet.reparam.L = 1;
    nets.encrpm.reparam.poolsize = numbatch;
    nets.decrpm.reparam.poolsize = numbatch;
    nets.priornet.reparam.poolsize = numbatch;
    
    lossnode = lossfun();
    
    %% main loop
    bestscore = Inf;
    if gpumode
        loss = zeros(numepoch, 1, 'single', 'gpuArray');
        labels = zeros(N, 1, 'single', 'gpuArray');
        z = zeros(nets.priornet.weight.J, N, 'single', 'gpuArray');
    else
        loss = zeros(numepoch, 1);
        labels = zeros(N, 1);
        z = zeros(nets.priornet.weight.J, N);
    end
    
    for epoch=1:numepoch
        tic;
        rndidx = randperm(N);
        nets.encrpm.reparam.init();
        nets.decrpm.reparam.init();
        nets.priornet.reparam.init();
        
        for batch=1:numbatch
            %profile on
            batchidx(1) = batchidx(2) + 1;
            batchidx(2) = batchidx(1) + batchsize - 1;
            x = data(:, rndidx(batchidx(1):batchidx(2)));
            
            % forward propagation
            loss(epoch) = loss(epoch) + fprop(x, nets, lossnode);
            assert(isnan(loss(epoch)) == 0, 'NaN is occured!');
            
            % clustering assignment
            [gam, zt] = posterior(x, nets.encnet, nets.encrpm, nets.priornet);
            [~, I] = max(gam);
            labels(rndidx(batchidx(1):batchidx(2)), 1) = I';
            z(:, rndidx(batchidx(1):batchidx(2))) = zt;
            
            % backward propagation
            bprop(nets, lossnode);
            
            % gradient checking
            %gradcheck(x, nets, lossnode);pause
            
            % update
            update(nets);
            
            %profile off
            %profile viewer
        end
        
        if loss(epoch) < bestscore
            bestscore = loss(epoch);
            fprintf('--current best score is updated to %e at epoch %d--\n', bestscore, epoch);
            bestprms = nets;
            
            timestamp = datestr(datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss'));
            savemodel(strcat(datafile, '_vde.mat'), bestprms, timestamp);
            save(strcat(datafile, '_vde_clustering_result.mat'), 'labels');
        end
        
        fprintf('epoch %d, loss: %e', epoch, loss(epoch));
        figure(1); plot(loss(1:epoch)); drawnow;
        figure(2); displabeling(data, labels);
        figure(3); dispcluster(z, labels, epoch);
        figure(4); disprecon(x, nets, lossnode, disprecon_helper);
        evalresult();
        drawnow();
        
        batchidx = batchidx .* 0;
        
        t = toc;
        fprintf(' [elapsed time %3.3f, %s]\n', t, datestr(datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss')));
    end
end

function savemodel(filename, bestprms, timestamp)
    netnames = fieldnames(bestprms);
    for i=1:length(netnames)
        nodenames = fieldnames(bestprms.(netnames{i}));
        
        for j=1:length(nodenames)
            bestprms.(netnames{i}).(nodenames{j}).refresh();
        end
    end
    
    save(filename, 'bestprms', 'timestamp');
end

function disprecon(x, nets, lossnode, helper)
    [~, output] = fprop(x, nets, lossnode);
    
    helper(x, output);
end

function defaultplot(x, output)
    I = 6;
    
    for i=1:I
        subplot(I, 1, i);
        plot(x(:, i)); hold on; 
        plot(output(:, i), 'm-.'); hold off;
    end
end

function dispcluster3D(rprsn, labels, epoch)
    elem= unique(labels);
    K = length(elem);
    
    hsv = [linspace(0.0, 0.9, K)', 0.8.*ones(K, 1), 0.75.*ones(K, 1)];
    c = hsv2rgb(hsv);
    
    for k=1:K
        idx = labels == elem(k);
        
        if size(rprsn, 1) > 2
            z = rprsn(3, idx);
        else
            z = zeros(length(find(idx == 1)), 1);
        end
        
        scatter3(rprsn(1, idx), rprsn(2, idx), z, 20, c(k, :)); hold on 
    end
    hold off
    title(epoch);
end

function displabeling3D(data, labels)
    elem= unique(labels);
    K = length(elem);
    
    hsv = [linspace(0.0, 0.9, K)', 0.8.*ones(K, 1), 0.75.*ones(K, 1)];
    c = hsv2rgb(hsv);
    
    for k=1:K
        idx = labels == elem(k);
        scatter3(data(1, idx), data(2, idx), data(3, idx), 20, c(k, :)); hold on
    end
    hold off
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