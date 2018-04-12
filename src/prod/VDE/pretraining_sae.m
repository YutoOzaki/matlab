function pretraining_sae
    %% load data
    load('testdata.mat');
    N = size(data, 2);
    D = size(data, 1);
    
    %% define networks
    numnode = [D, 30, 10];
    
    encnet = struct(...
        'l1con', lineartrans(numnode(2), numnode(1), rmsprop(0.9, 1e-2, 1e-8)),...
        'l1act', tanhtrans(),...
        'l2con', lineartrans(numnode(3), numnode(2), rmsprop(0.9, 1e-2, 1e-8)),...
        'l2act', tanhtrans()...
        );
    
    decnet = struct(...
        'l2con', lineartrans(numnode(2), numnode(3), rmsprop(0.9, 1e-2, 1e-8)),...
        'l2act', tanhtrans(),...
        'l1con', lineartrans(numnode(1), numnode(2), rmsprop(0.9, 1e-2, 1e-8)),...
        'l1act', identitymap()...
        );
    
    names = fieldnames(encnet);
    for i=1:length(names)
        encnet.(names{i}).init();
    end
    
    names = fieldnames(decnet);
    for i=1:length(names)
        decnet.(names{i}).init();
    end
    
    %% define configuration
    numepoch = 100;
    batchsize = 50;
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
            input = x;
            names = fieldnames(encnet);
            for i=1:length(names)
                input = encnet.(names{i}).forwardprop(input);
            end

            names = fieldnames(decnet);
            for i=1:length(names)
                input = decnet.(names{i}).forwardprop(input);
            end
            
            L = 0.5.*sum((x - input).^2);
            
            loss(epoch) = loss(epoch) + sum(L)/batchsize;
            
            % backward propagation
            delta = -(x - input);
            
            names = flipud(fieldnames(decnet));
            for i=1:length(names)
                delta = decnet.(names{i}).backwardprop(delta);
            end
            
            names = flipud(fieldnames(encnet));
            for i=1:length(names)
                delta = encnet.(names{i}).backwardprop(delta);
            end
            
            % gradient checking
            %gradcheck(x, encnet, decnet);
            
            % update
            update(struct('encnet', encnet, 'decnet', decnet));
        end
        
        figure(1); plot(loss(1:epoch)); drawnow;
        fprintf('epoch %d, loss: %e\n', epoch, loss(epoch));
        
        figure(2);
        I = min(batchsize, 5);
        for i=1:I
            subplot(I,1,i); plot(x(:, i)); hold on; plot(input(:, i), 'm-.'); hold off;
        end
        drawnow;
        
        batchidx = batchidx .* 0;
    end
    
    save('pretrained.mat', 'encnet', 'decnet');
end

function gradcheck(x, encnet, decnet)
    eps = 1e-6;
    f = zeros(2, 1);
    d = zeros(2, 1);
    batchsize = size(x, 2);
    
    nets = struct('encnet', encnet, 'decnet', decnet);
    
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
                input = x;
                names = fieldnames(encnet);
                for ii=1:length(names)
                    input = encnet.(names{ii}).forwardprop(input);
                end

                names = fieldnames(decnet);
                for ii=1:length(names)
                    input = decnet.(names{ii}).forwardprop(input);
                end
                f(1) = sum(0.5.*sum((x - input).^2))/batchsize;

                prm(a, b) = val - eps;
                nets.(netnames{i}).(nodenames{j}).prms.(prmnames{k}) = prm;
                input = x;
                names = fieldnames(encnet);
                for ii=1:length(names)
                    input = encnet.(names{ii}).forwardprop(input);
                end

                names = fieldnames(decnet);
                for ii=1:length(names)
                    input = decnet.(names{ii}).forwardprop(input);
                end
                f(2) = sum(0.5.*sum((x - input).^2))/batchsize;

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