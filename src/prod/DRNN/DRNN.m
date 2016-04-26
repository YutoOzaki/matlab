function DRNN()
    %% load data
    %load testdata
    load('C:\Users\yuto\Documents\MATLAB\data\gtzan\gtzanMFCC.mat');
    dim = size(trainMat,1);
    N_train = size(trainMat,2);
    T = size(trainMat,3);
    N_test = size(testMat,2);
    
    %% set hyperparameters
    L = 2;
    epochs = 15;
    patch = 120;
    batchSize = 100;
    hid = {256 10};
    ongpu = false;
    gcheck = false;
    m2o = 'mean'; % 'mean' or 'max';
    
    unitNum = [dim,hid];
    nnet = cell(1,L);
    
    if strcmp(m2o,'max')
        manyToOne = @(x) max(x(:,:,end));
    elseif strcmp(m2o,'mean')
        manyToOne = @(x) max(mean(x,3));
    end
    
    %% data formatting
    prediction = zeros(unitNum{end}, N_test, T);
    labels = testLabel(1:patch:end);
    
    trainLabelVector = oneHotVectorLabel(trainLabel,hid{end},T);
    testLabelVector = oneHotVectorLabel(testLabel,hid{end},T);
    
    trainLabel = repmat(trainLabel', 1, 2);
    testLabel = repmat(testLabel', 1, 2);
    
    if ongpu
        trainMat = gpuArray(trainMat);
        trainLabelVector = gpuArray(trainLabelVector);
        trainLabel =gpuArray(trainLabel);
    
        testMat = gpuArray(testMat);
        testLabelVector = gpuArray(testLabelVector);
        testLabel = gpuArray(testLabel);
    end
    
    %% instance nerual nets
    for l=1:L-1
        nnet{l} = GRU();
        nnet{l}.initLayer(unitNum{l},unitNum{l+1},T,batchSize);
        nnet{l}.optimization('rmsProp',[0.01 0.9 1e-8]);
        nnet{l}.onGPU(ongpu);
    end
    
    nnet{L} = SoftmaxLayer();
    nnet{L}.initLayer(unitNum{L},unitNum{L+1},T,batchSize);
    nnet{L}.optimization('rmsProp',[0.01 0.9 1e-8]);
    nnet{L}.onGPU(ongpu);
    
    %% gradient checking
    batchNumCnt = 1;
    if gcheck
        gcloop = 1;
        batchNum = N_train/batchSize;
        
        reLog = cell(1,L);
        for l=1:L
            reLog{1,l} = zeros(3,nnet{l}.prmNum,batchNum);
        end
    else
        gcloop = L+1;
    end
    
    %% main loop
    for i=1:epochs
        fprintf('\nepoch %d\n', i);
        rndidx = randperm(N_train);
        
        %% training
        tic;
        for n=1:batchSize:N_train
            idx = n:n+batchSize-1;
            
            batchLog = sprintf('(batch %d - %d)',idx(1),idx(end));
            fprintf(batchLog);
            idx = rndidx(idx);
            
            input = trainMat(:,idx,:);
            for l=1:L
                input = nnet{l}.fprop(input);
            end
            
            delta = input - trainLabelVector(:,idx,:);
            for l=L:-1:1
                delta = nnet{l}.bprop(delta);
            end
            
            for l=gcloop:L
                meps = eps^(1/3);
                reBuf = reLog{1,l};
                
                for k=1:nnet{l}.prmNum
                    val = nnet{l}.prms{k}(1,1);
                    h = max(abs(val),1) * meps;
                    
                    nnet{l}.prms{k}(1,1) = val + h;
                    input = trainMat(:,idx,:);
                    for p=1:L
                        input = nnet{p}.fprop(input);
                    end
                    dy1 = sum(sum(trainLabelVector(:,idx,:).*log(input),3),1);

                    nnet{l}.prms{k}(1,1) = val - h;
                    input = trainMat(:,idx,:);
                    for p=1:L
                        input = nnet{p}.fprop(input);
                    end
                    dy2 = sum(sum(trainLabelVector(:,idx,:).*log(input),3),1);
                    
                    nnet{l}.prms{k}(1,1) = val;

                    dt1 = mean((-dy1 + dy2)./(2*h));
                    dt2 = nnet{l}.gprms{k}(1,1);
                    relativeError = abs(dt1 - dt2)/max(abs(dt1),abs(dt2));
                    reBuf(:,k,batchNumCnt) = [relativeError,dt1,dt2];
                end
                
                reLog{1,l} = reBuf;
            end
            batchNumCnt = batchNumCnt + 1;
            
            for l=1:L
                nnet{l}.update();
            end
            
            for c=1:length(batchLog)
                fprintf('\b');
            end
        end
        t = toc;
        fprintf('elapsed time %3.3f (training)\n', t);
        
        for l=gcloop:L
            subplot(L,1,l);plot(log10(squeeze(reLog{l}(1,:,:)))');ylim([-10 0]);
            drawnow
        end
        batchNumCnt = 1;
        
        %% inference on training data set
        tic;
        loss_training = 0;
        for n=1:batchSize:N_train
            idx = n:n+batchSize-1;
            
            input = trainMat(:,idx,:);
            for l=1:L
                input = nnet{l}.fprop(input);
            end
            
            [~,mind] = manyToOne(input);
            trainLabel(idx,1) = mind;
            
            loss_training = loss_training + trainLabelVector(:,idx,:).*log(input);
        end
        loss_training = sum(sum(sum(loss_training)));
        t = toc;
        
        result_train = length(find((trainLabel(:,1) - trainLabel(:,2)) == 0));
        fprintf('result: %3.3f%%, %5.3f (training set)\n',100*(result_train/N_train),loss_training);
        fprintf('elapsed time %3.3f (inference on training data set)\n', t);
        
        %% inference on test data set
        tic;
        loss_test = 0;
        for n=1:batchSize:N_test
            idx = n:n+batchSize-1;
            
            input = testMat(:,idx,:);
            for l=1:L
                input = nnet{l}.fprop(input);
            end
            
            [~,mind] = manyToOne(input);
            testLabel(idx,1) = mind;
            prediction(:,idx,:) = input;
            
            loss_test = loss_test + testLabelVector(:,idx,:).*log(input);
        end
        loss_test = sum(sum(sum(loss_test)));
        t = toc;
        
        %% apply voting scheme for sub-sampled data
        vposT = votingSummary(patch, testLabel, unitNum{end});
        cposT = confidenceScheme(prediction, patch, labels);
        %vposT = -1; cposT = -1;
        
        %% print out result
        result_test = length(find((testLabel(:,1) - testLabel(:,2)) == 0));
        fprintf('result: %3.3f%%, %5.3f (test set)\n',100*(result_test/N_test),loss_test);
        fprintf('Classification accuracy (voting): %3.3f\n', vposT);
        fprintf('Classification accuracy (confidence): %3.3f\n', cposT);
        fprintf('elapsed time %3.3f (inference on test data set)\n', t);
    end
end