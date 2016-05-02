function DRNN()
    %% load data
    load testdata
    %load('C:\Users\yuto\Documents\MATLAB\data\gtzan\gtzanMFCC.mat');
    dim = size(trainMat,1);
    N_train = size(trainMat,2);
    T = size(trainMat,3);
    N_test = size(testMat,2);
    
    %% set hyperparameters
    epochs = 15;
    patch = 120;
    batchSize = 3;
    hid = {20 3};
    ongpu = false;
    BN = false;
    gcheck = false;
    m2o = 'max'; % 'mean' or 'max';
    
    L = length(hid);
    
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
        nnet{l} = BLSTM();
        nnet{l}.initLayer(unitNum{l},unitNum{l+1},T,batchSize,BN);
        nnet{l}.optimization('rmsProp',[0.01 0.9 1e-8]);
        %nnet{l}.optimization('adaDelta',[0.95 1e-6]);
        %nnet{l}.optimization('adaGrad',[0.1 1e-8]);
        %nnet{l}.optimization('adam',[1e-3 0.9 0.999 1e-7]);
        nnet{l}.onGPU(ongpu);
    end
    
    nnet{L} = BSoftmaxLayer();
    nnet{L}.initLayer(unitNum{L},unitNum{L+1},T,batchSize,BN);
    nnet{L}.optimization('rmsProp',[0.01 0.9 1e-8]);
    %nnet{L}.optimization('adaDelta',[0.95 1e-6]);
    %nnet{L}.optimization('adaGrad',[0.1 1e-8]);
    %nnet{L}.optimization('adam',[1e-3 0.9 0.999 1e-7]);
    nnet{L}.onGPU(ongpu);
    
    %% gradient checking
    gchecker = GradientChecker(gcheck, L, N_train/batchSize ,nnet);
    
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
            
            gchecker.gradientChecking(nnet,trainMat(:,idx,:),trainLabelVector(:,idx,:));
            
            for l=1:L
                nnet{l}.update();
            end
            
            for c=1:length(batchLog)
                fprintf('\b');
            end
        end
        t = toc;
        fprintf('elapsed time %3.3f (training)\n', t);
        gchecker.disp();
        
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
        %vposT = votingSummary(patch, testLabel, unitNum{end});
        %cposT = confidenceScheme(prediction, patch, labels);
        vposT = -1; cposT = -1;
        
        %% print out result
        result_test = length(find((testLabel(:,1) - testLabel(:,2)) == 0));
        fprintf('result: %3.3f%%, %5.3f (test set)\n',100*(result_test/N_test),loss_test);
        fprintf('Classification accuracy (voting): %3.3f\n', vposT);
        fprintf('Classification accuracy (confidence): %3.3f\n', cposT);
        fprintf('elapsed time %3.3f (inference on test data set)\n', t);
    end
end