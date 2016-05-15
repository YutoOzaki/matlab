function bestResult = DRNN(hprms)
    showUnitNum(hprms.unitNum);
    disp(hprms);
    
    %% set hyperparameters
    batchSize = hprms.batchSize;
    unitNum = hprms.unitNum;
    T = hprms.T;
    
    epochs = hprms.epochs;
    noUpdatSpanCriterion = hprms.noUpdatSpanCriterion;
    longTerm = hprms.longTerm;
    gcheck = false;
    
    L = length(hprms.type);
    nnet = cell(L,1);
    
    scheme = {'mean', 'max', 'max rate'};
    bestResult = zeros(length(scheme), 2);
    
    dataSet = hprms.dataSet;
    %initT = @(a,b) 1;
    %initT = @(T, T_total) randi(mod(T_total, T));
    initT = @(T, a) randi(T);
    
    %% create neural net
    for l=1:L
        eval(strcat('nnet{l} = ', hprms.type{l}, '()', ';'));
        nnet{l}.initLayer(unitNum{l}, unitNum{l+1}, T, batchSize, hprms.BN(l), hprms.dropRate(l), hprms.clipping(l));
        nnet{l}.optimization(hprms.gdoa, hprms.gdoaPrm);
        nnet{l}.onGPU(hprms.gpumode);
    end
    
    %% gradient checking
    gchecker = GradientChecker(gcheck, L, floor(dataSet{1}.N/batchSize), nnet);
    
    %% main loop
    for i=1:epochs
        fprintf('\nepoch %d\n', i);
        
        %% training
        tic;
        dataSet{1}.shuffle();
        while dataSet{1}.setNextIndex();
            batchLog = sprintf('(batch %d - %d)', dataSet{1}.index(1), dataSet{1}.index(end));
            fprintf(batchLog);
            
            batchData = dataSet{1}.getFeature();
            labelVector = dataSet{1}.label2vec(dataSet{1}.getLabel(), unitNum{end}, T);
            T_total = size(batchData, 3);
            
            t = initT(T, T_total);
            if longTerm
                for l=1:L
                    nnet{l}.resetStates();
                end
            end
            
            while t+T-1 <= T_total
                input = batchData(:,:,t:t+T-1);
                for l=1:L
                    if longTerm
                        nnet{l}.continueStates();
                    end
                    nnet{l}.createDropMask();
                    input = nnet{l}.fprop(input);
                end
                
                delta = input - labelVector;
                for l=L:-1:1
                    delta = nnet{l}.bprop(delta);
                end

                gchecker.gradientChecking(nnet, batchData(:,:,t:t+T-1), labelVector);

                for l=1:L
                    nnet{l}.update();
                end
                
                t = t + T;
            end
            
            fprintf(repmat('\b',1,length(batchLog)));
        end
        t = toc;
        fprintf('elapsed time %3.3f (training)\n', t);
        
        %% inference
        figure(1);
        for j=1:3
            bufResult = inferenceMode(dataSet{j}, nnet, unitNum{end}, longTerm, j);
        end
        
        fprintf('--Best classification results--\n');
        for j=1:length(scheme)
            if bestResult(j,1) < bufResult(j)
                bestResult(j,:) = [bufResult(j) i];
            end
            
            fprintf(' %3.3f%% at epoch %d (%s)\n', bestResult(j,1), bestResult(j,2), scheme{j});
        end
        
        [~, idx] = max(bestResult(:,1));
        if i - bestResult(idx,2) >= noUpdatSpanCriterion
            break;
        end
    end
    
    fprintf('\n**finished**\n');
    for i=1:length(scheme)
        fprintf('%3.3f%% at epoch %d (%s)\n', bestResult(i,1), bestResult(i,2), scheme{i});
    end
end

function showUnitNum(unitNum)
    fprintf('units: %d', unitNum{1});
    
    for i=2:length(unitNum)
        fprintf(' - ');
        
        if iscell(unitNum{i})
            fprintf('(%d', unitNum{i}{1});
            
            for j=2:length(unitNum{i})
                fprintf(' - %d', unitNum{i}{j});
            end
            
            fprintf(')');
        else
            fprintf('%d', unitNum{i});
        end
    end
    
    fprintf('\n');
end