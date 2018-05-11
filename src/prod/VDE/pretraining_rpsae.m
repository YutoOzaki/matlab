function pretraining_rpsae(datafile, h, act, L, numepoch, gpumode, drawrecon)
    switch nargin
        case 6
            drawrecon = @defaultplot;
    end
    
    %% load data
    load(datafile);
    
    if gpumode
        data = gpuArray(cast(data, 'single'));
    end
    
    N = size(data, 2);
    D = size(data, 1);
    
    %% define configuration
    batchsize = 128;
    numbatch = floor(N / batchsize);
    batchidx = zeros(2, 1);
    
    %% define networks
    numnode = [D h];
    weidec = 1e-3;
    
    encnet = struct();
    for l=1:(length(numnode) - 2)
        nodename = strcat('l', num2str(l), 'con');
        encnet.(nodename) = lineartrans(numnode(l + 1), numnode(l), adam(0.9, 0.999, 1e-3, 1e-8, 'desc'), weidec, gpumode);
        nodename = strcat('l', num2str(l), 'act');
        encnet.(nodename) = copy(act{l});
    end
    
    encrpm = struct(...
        'mu', lineartrans(numnode(end), numnode(end - 1), adam(0.9, 0.999, 1e-3, 1e-8, 'desc'), weidec, gpumode),...
        'lnsigsq', lineartrans(numnode(end), numnode(end - 1), adam(0.9, 0.999, 1e-3, 1e-8, 'desc'), weidec, gpumode),...
        'exp', exptrans(),...
        'reparam', reparamtrans(numnode(end), L, numbatch, gpumode)...
        );
    
    decnet = struct();
    for l=length(numnode):-1:3
        nodename = strcat('l', num2str(l), 'con');
        decnet.(nodename) = lineartrans(numnode(l - 1), numnode(l), adam(0.9, 0.999, 1e-3, 1e-8, 'desc'), weidec, gpumode);
        nodename = strcat('l', num2str(l), 'act');
        decnet.(nodename) = copy(act{l - 2});
    end
    
    decrpm = struct(...
        'mu', lineartrans(numnode(1), numnode(2), adam(0.9, 0.999, 1e-3, 1e-8, 'desc'), weidec, gpumode),...
        'lnsigsq', lineartrans(numnode(1), numnode(2), adam(0.9, 0.999, 1e-3, 1e-8, 'desc'), weidec, gpumode),...
        'exp', exptrans(),...
        'reparam', reparamtrans(numnode(1), L, numbatch, gpumode)...
        );
    
    nets = struct('encnet', encnet, 'encrpm', encrpm, 'decnet', decnet, 'decrpm', decrpm);
    netnames = fieldnames(nets);
    for i=1:length(netnames)
        nodenames = fieldnames(nets.(netnames{i}));
        
        for j=1:length(nodenames)
            nets.(netnames{i}).(nodenames{j}).init();
        end
    end
    
    %% main loop
    bestprms = nets;
    bestscore = Inf;
    if gpumode
        loss_hist = zeros(numepoch, 1, 'single', 'gpuArray');
    else
        loss_hist = zeros(numepoch, 1);
    end
            
    devinf = gpuDevice;
    memrec = zeros(numepoch*numbatch, 1);
    ii = 1;
    memrec(ii) = devinf.AvailableMemory;
    for epoch=1:numepoch
        %profile on
        tic;
        rndidx = randperm(N);
        nets.encrpm.reparam.init();
        nets.decrpm.reparam.init();
        
        for batch=1:numbatch
            batchidx(1) = batchidx(2) + 1;
            batchidx(2) = batchidx(1) + batchsize - 1;
            x = data(:, rndidx(batchidx(1):batchidx(2)));
            
            % forward propagation
            [loss, output] = fprop(x, nets);
            loss_hist(epoch) = loss_hist(epoch) + loss;
            
            % backward propagation
            delta = -(x - output);
            bprop(delta, nets);
            
            % gradient checking
            %gradcheck(x, nets); pause
            
            % update
            update(nets);
            
            ii = ii + 1;
            memrec(ii) = devinf.AvailableMemory;
        end
        
        if loss_hist(epoch) < bestscore
            bestscore = loss_hist(epoch);
            fprintf('--current best score is updated to %e at epoch %d--\n', bestscore, epoch);
            bestprms = nets;
        end
       
        figure(1); plot(loss_hist(1:epoch)); drawnow;
        fprintf('epoch %d, loss: %e', epoch, loss_hist(epoch));
        
        figure(2); drawrecon(x, output);
        figure(3); subplot(2,1,1); plot(memrec(1:ii)); subplot(2,1,2); plot(memrec((epoch-1)*numbatch+1:ii));
        drawnow;
        
        batchidx = batchidx .* 0;
        
        t = toc;
        fprintf(' [elapsed time %3.3f, %s]\n', t, datestr(datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss')));
        %profile off
        %profile viewer
    end
    
    savemodel(strcat(datafile, '_pretrained.mat'), bestprms);
end

function defaultplot(x, output)
    I = 6;
    
    for i=1:I
        subplot(I, 1, i);
        plot(x(:, i)); hold on; 
        plot(output(:, i), 'm-.'); hold off;
    end
end

function savemodel(filename, bestprms)
    netnames = fieldnames(bestprms);
    for i=1:length(netnames)
        nodenames = fieldnames(bestprms.(netnames{i}));
        
        for j=1:length(nodenames)
            bestprms.(netnames{i}).(nodenames{j}).refresh();
        end
    end
    
    save(filename, 'bestprms');
end

function bprop(delta, nets)
    L = nets.decrpm.reparam.L;
    delta = nets.decrpm.reparam.backwardprop(repmat(delta, [1, 1, L])./L);
    dxmu = nets.decrpm.mu.backwardprop(delta.mu);
    dxsig = nets.decrpm.exp.backwardprop(delta.sig);
    dlnxsig = nets.decrpm.lnsigsq.backwardprop(dxsig);
    delta = dxmu + dlnxsig;
    
    names = flipud(fieldnames(nets.decnet));
    for i=1:length(names)
        delta = nets.decnet.(names{i}).backwardprop(delta);
    end

    L = nets.encrpm.reparam.L;
    delta = nets.encrpm.reparam.backwardprop(repmat(delta, [1, 1, L])./L);
    dxmu = nets.encrpm.mu.backwardprop(delta.mu);
    dxsig = nets.encrpm.exp.backwardprop(delta.sig);
    dlnxsig = nets.encrpm.lnsigsq.backwardprop(dxsig);
    delta = dxmu + dlnxsig;
    
    names = flipud(fieldnames(nets.encnet));
    for i=1:length(names)
        delta = nets.encnet.(names{i}).backwardprop(delta);
    end
end

function [loss, output] = fprop(x, nets)
    batchsize = size(x, 2);

    input = x;
    names = fieldnames(nets.encnet);
    for i=1:length(names)
        input = nets.encnet.(names{i}).forwardprop(input);
    end
    
    mu = nets.encrpm.mu.forwardprop(input);
    sigsq = nets.encrpm.exp.forwardprop(nets.encrpm.lnsigsq.forwardprop(input));
    z = nets.encrpm.reparam.forwardprop(struct('mu', mu, 'sig', sigsq));

    z = mean(z, 3);
    
    input = z;
    names = fieldnames(nets.decnet);
    for i=1:length(names)
        input = nets.decnet.(names{i}).forwardprop(input);
    end

    mu = nets.decrpm.mu.forwardprop(input);
    sigsq = nets.decrpm.exp.forwardprop(nets.decrpm.lnsigsq.forwardprop(input));
    output = nets.decrpm.reparam.forwardprop(struct('mu', mu, 'sig', sigsq));
    
    output = mean(output, 3);
    
    loss = 0.5.*sum((x - output).^2);
    
    wreg = 0;
    netnames = fieldnames(nets);
    for i=1:length(netnames)
        nodenames = fieldnames(nets.(netnames{i}));
        
        for j=1:length(nodenames)
            prmnames = fieldnames(nets.(netnames{i}).(nodenames{j}).prms);
            
            for k=1:length(prmnames)
                if strcmp('W', prmnames{k})
                    wreg = wreg + 0.5*nets.(netnames{i}).(nodenames{j}).weidec.*sum(sum(nets.(netnames{i}).(nodenames{j}).prms.(prmnames{k}).^2));
                end
            end
        end
    end
    
    loss = sum(loss)/batchsize + wreg;
end

function gradcheck(x, nets)
    eps = 1e-6;
    f = zeros(2, 1, class(x));
    d = zeros(2, 1, class(x));
    
    netnames = fieldnames(nets);
    for i=1:length(netnames)
        nodenames = fieldnames(nets.(netnames{i}));

        for j=1:length(nodenames)
            prmnames = fieldnames(nets.(netnames{i}).(nodenames{j}).prms);

            for k=1:length(prmnames)
                prm = nets.(netnames{i}).(nodenames{j}).prms.(prmnames{k});

                m = size(prm, 1);
                n = size(prm, 2);
                a = randi(m);
                b = randi(n);
                val = prm(a, b);

                prm(a, b) = val + eps;
                nets.(netnames{i}).(nodenames{j}).prms.(prmnames{k}) = prm;
                f(1) = fprop(x, nets);
                
                prm(a, b) = val - eps;
                nets.(netnames{i}).(nodenames{j}).prms.(prmnames{k}) = prm;
                f(2) = fprop(x, nets);

                d(1) = (f(1) - f(2))/(2*eps);
                d(2) = nets.(netnames{i}).(nodenames{j}).grad.(prmnames{k})(a, b);

                re = abs(d(1) - d(2))/max(abs(d(1)), abs(d(2)));
                fprintf('%s.%s.%s, %e, %e, %e, %e\n', netnames{i}, nodenames{j}, prmnames{k}, val, d(1), d(2), re);

                prm(a, b) = val;
                nets.(netnames{i}).(nodenames{j}).prms.(prmnames{k}) = prm;
            end
        end
    end
end