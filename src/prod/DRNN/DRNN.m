function DRNN()
    %% set hyperparameters
    batchSize = 100;
    unitNum = {40 128 10};
    T = 80;
    BN = [false false];
    dropRate = [0.0 0.0];
    
    epochs = 50;
    noUpdatSpanCriterion = 10;
    longTerm = false;
    ongpu = false;
    gcheck = false;
    
    L = length(unitNum) - 1;
    nnet = cell(1,L);
    
    scheme = {'mean', 'max', 'max rate'};
    bestResult = zeros(length(scheme), 2);
    
    %% create dataset
    N = [900 0 100]; % training, validation and test
    getDataSet = @() prepGTZAN(batchSize, N);
    %getDataSet = @() prepTestData(batchSize, 'testdata.mat');
    %getDataSet = @() prepTestData(batchSize, 'C:\Users\yuto\Documents\MATLAB\data\gtzan\gtzanMFCC.mat');
    dataSet = getDataSet();
    
    %% instance nerual nets
    for l=1:L-1
        nnet{l} = LSTM();
        nnet{l}.initLayer(unitNum{l},unitNum{l+1},T,batchSize,BN(l),dropRate(l));
        nnet{l}.optimization('rmsProp',[0.01 0.9 1e-8]);
        %nnet{l}.optimization('adaDelta',[0.95 1e-6]);
        %nnet{l}.optimization('adaGrad',[0.1 1e-8]);
        %nnet{l}.optimization('adam',[1e-3 0.9 0.999 1e-7]);
        nnet{l}.onGPU(ongpu);
    end
    
    nnet{L} = SoftmaxLayer();
    nnet{L}.initLayer(unitNum{L},unitNum{L+1},T,batchSize,BN(L),dropRate(L));
    nnet{L}.optimization('rmsProp',[0.01 0.9 1e-8]);
    %nnet{L}.optimization('adaDelta',[0.95 1e-6]);
    %nnet{L}.optimization('adaGrad',[0.1 1e-8]);
    %nnet{L}.optimization('adam',[1e-3 0.9 0.999 1e-7]);
    nnet{L}.onGPU(ongpu);
    
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
            
            t_mod = mod(T_total,T);
            if t_mod == 0
                t = 1;
            else
                t = randi(t_mod);
            end
            
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
end