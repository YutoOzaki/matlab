function DRNN(name,hprms)
    %% home directory
    home = userpath;
    home = home(1:end-1);
    
    %% enable to collect log
    diary off
    timeStamp = clock;
    timeStr = '';
    for i=1:length(timeStamp)-1
        if i~= 1
            timeBuf = num2str(timeStamp(i),'%02i');
        else
            timeBuf = num2str(timeStamp(i));
        end
        timeStr = strcat(timeStr,timeBuf);
    end
    diary(strcat(home,'/logs/',name,'/',name,'_',timeStr,'.txt'));
    clear timeStamp timeBuf i

    %% Read and show hyper-parameters    
    nhid         = hprms.nhid;
    units        = hprms.units;
    optimization = hprms.optimization;
    batchSize    = hprms.batchSize;
    baselr       = hprms.baselr;
    epochs       = hprms.epochs;
    drate        = hprms.drate;
    droprate     = hprms.droprate;
    gcheck       = hprms.gcheck;
    dataPrep     = hprms.dataPrep;
    patch        = hprms.patch;
    
    disp(hprms);
    disp(dataPrep);
    clear hprms

    %% load data
    [trainMat,testMat,trainLabel,testLabel] = dataPrep();
    samples = size(trainMat,2); testSamples = size(testMat,2); T = size(trainMat,3);

    %% validation
    assert(length(nhid)==length(units),...
        'number of units must agree');
    assert(length(unique(trainLabel))==nhid{end},...
        'number of training class must agree');
    assert(length(unique(testLabel))==nhid{end},...
        'number of testing class must agree');
    assert(size(trainMat,1)==size(testMat,1),...
        'number of dimension must agree');
    assert(mod(samples,batchSize)==0,...
        'batch size should be divisible with the training sample size');
    assert(mod(testSamples,batchSize)==0,...
        'batch size should be divisible with the testing sample size');
    if strcmp(optimization,'rmsprop') || strcmp(optimization,'adadelta')
        assert(exist('drate','var')==1,...
            'variable drate is necessary for rmsprop and adadelta');
    end
    if sum(ismember(units,'bmcell')) > 0
        assert(sum(ismember(units,'mcell'))==0,...
            'bidirectional units should not be used with directional units');
    end
    assert(length(droprate)==length(nhid),...
        'each layer should have dropout rate')
    
    for i=1:patch:samples
        assert(length(unique(trainLabel(i:i+patch-1)))==1,...
            'patch data are mixed');
    end
    
    for i=1:patch:testSamples
        assert(length(unique(testLabel(i:i+patch-1)))==1,...
            'patch data are mixed');
    end

    %% set variables
    L = length(nhid);

    dims = {size(trainMat,1)};
    fprops = cell(1,L);
    prms = cell(1,L);
    auxPrms = cell(1,L);
    bprops = cell(1,L);
    inputs = cell(1,L);
    updateFun = cell(1,3);
    gprms = cell(1,L);
    dropFun = cell(1,L);
    weiComp = cell(1,L);

    if strcmp(optimization,'rmsprop')==1
        rmsPrms = cell(1,L);

        updateFun{1,1} = @rmsProp;
        updateFun{1,3} = [drate eps];
    elseif strcmp(optimization,'adadelta')==1
        adadMat = cell(1,L);

        updateFun{1,1} = @AdaDelta;
        updateFun{1,3} = [drate 1e-7];
    elseif strcmp(optimization,'vanilla')==1
        lrmin = 1e-6;
        lrdec = 0.99;

        vanillaPrms = cell(1,L);
        for l=1:L, vanillaPrms{l} = [lrdec 1.0]; end

        updateFun{1,1} = @vanillaSGD;
        updateFun{1,2} = vanillaPrms;
        updateFun{1,3} = lrmin;
    end

    for l=1:L
        dims{1,l+1} = nhid{l};

        vis = dims{l};
        hid = dims{l+1};

        if strcmp(units{l},'mcell')
            prm = initLSTMPrms(vis,hid);
            prms{l} = prm;
            auxPrms{l} = initLSTMState(hid,batchSize,T);

            fprops{l} = @fpropLSTM;
            bprops{l} = @bpropLSTM;
        elseif strcmp(units{l},'bmcell')
            assert(length(hid)==2,'bidirectional unit should have two hidden cells');
            if length(vis)==1, vis = [vis vis]; end

            prm = [initLSTMPrms(vis(1),hid(1)) initLSTMPrms(vis(2),hid(2))];
            prms{l} = prm;
            auxPrms{l} = [initLSTMState(hid(1),batchSize,T) initLSTMState(hid(2),batchSize,T)];

            fprops{l} = @fpropBLSTM;
            bprops{l} = @bpropBLSTM;
        elseif strcmp(units{l},'lp')
            assert(length(hid)==2,'Lp unit should have two hidden layers');

            prm = initLpPrms(vis,hid);
            prms{l} = prm;

            fprops{l} = @fpropLpUnit;
            bprops{l} = @bpropLpUnit;
        elseif strcmp(units{l},'smax')
            if strcmp(units{l-1},'bmcell')
                prm1 = initPrms(vis(1),hid);
                prm2 = initPrms(vis(2),hid);
                prm = [prm1 prm2{1}];
                prms{l} = prm;

                fprops{l} = @fpropBSoftmax;
                bprops{l} = @bpropBSoftmax;
                clear prm1 prm2
            else
                prm = initPrms(vis,hid);
                prms{l} = prm;

                fprops{l} = @fpropSoftmax;
                bprops{l} = @bpropSoftmax;
            end
        end

        if strcmp(optimization,'rmsprop')==1,
            rmsPrms{l} = cellfun(@(x) x.*0,prm,'UniformOutput',false);
        elseif strcmp(optimization,'adadelta')==1,
            buf = cellfun(@(x) x.*0,prm,'UniformOutput',false);
            adadMat{l} = {buf buf};
        end
    end

    if strcmp(optimization,'rmsprop')==1
        updateFun{1,2} = rmsPrms;
    elseif strcmp(optimization,'adadelta')==1
        updateFun{1,2} = adadMat;
    end

    if sum(ismember(units,'bmcell')) > 0
        expand = @(x) {x x};
        for i=1:L
            dropFun{i} = @(x) {dropoutMask(x{1},droprate(i)) dropoutMask(x{2},droprate(i))};
            weiComp{i} = @(x) {x{1}.*droprate(i) x{2}.*droprate(i)};
        end
    else
        expand = @(x) x;
        for i=1:L
            dropFun{i} = @(x) dropoutMask(x,droprate(i));
            weiComp{i} = @(x) x.*droprate(i);
        end
    end

    trainResults = zeros(samples,2);
    trainResults(:,2) = trainLabel;

    testResults = zeros(testSamples,2);
    testResults(:,2) = testLabel;

    trainLabel = oneHotVectorLabel(trainLabel,nhid{end},T);
    testLabel = oneHotVectorLabel(testLabel,nhid{end},T);

    [~,midx] = max(trainLabel(:,:,T));
    assert(sum(trainResults(:,2)-midx')==0,'check training label');
    [~,midx] = max(testLabel(:,:,T));
    assert(sum(testResults(:,2)-midx')==0,'check test label');

    if gcheck == true
        batchNum = samples/batchSize;
        gcloop = 1;
        rerror = cell(1,L);
        for l=1:L
            rerror{l} = zeros(length(prms{l}),3,batchNum);
        end
        tol = 1e-3;
    else
        gcloop = L + 1;
    end

    preResult = 0;

    %% memory info
    whos;
    S = whos;
    totalMem = 0;
    for i=1:length(S)
        totalMem = totalMem + S(i).bytes;
    end
    fprintf('Total amount of memory(MB): %3.3f\n\n',totalMem/(1024^2));

    %% main loop
    for ep=1:epochs
        fprintf('epoch %d: mini-batch  ',ep);
        for i=1:length(num2str(batchSize)), fprintf(' '); end
        tic;
        rndidx = randperm(samples);
        batchNum = 0;

        for n=1:batchSize:samples
            %% start of training
            idx = rndidx(n:n+batchSize-1);
            for i=1:length(num2str(n-2))+1, fprintf('\b'); end
            fprintf('%d~',n);

            %% forward propagation       
            input = trainMat(:,idx,:);
            input = expand(input);
            for l=1:L
                input = dropFun{l}(input);
                inputs{l} = input;
                [input,states] = fprops{l}(input,prms{l},T,auxPrms{l});
                auxPrms{l} = states;
            end

            %% backward propagation
            delta = input - trainLabel(:,idx,:);
            for l=L:-1:1
                [delta,gprm] = bprops{l}(delta,inputs{l},prms{l},T,auxPrms{l});
                gprms{l} = gprm;
            end

            %% monitor norms of gradient
            figure(4);
            for l=L:-1:1       
                subplot(L,1,l);plot(gradNorm(gprms{l}));
            end
            drawnow

            %% gradient checking
            batchNum = batchNum + 1;
            for l=gcloop:L
                rerror{l}(:,:,batchNum) ...
                    = gradientChecking(l,gprms{l},trainLabel(:,idx,T),eps,trainMat(:,idx,:),...
                    fprops,expand,weiComp,prms,auxPrms,T);
            end

            %% update parameterrs
            for l=1:L                
                [prm,updatePrm] = updateFun{1}(prms{l},gprms{l},baselr,updateFun{2}{l},updateFun{3});
                updateFun{2}{l} = updatePrm;
                prms{l} = prm;
            end
        end
        fprintf(', ellapsed time %4.4f\n',toc');

        %% training set
        fprintf(' training set:\t'); tic;
        output = fpropBatch(trainMat,batchSize,fprops,expand,weiComp,prms,auxPrms,T);
        [~,midx] = max(output(:,:,T));
        trainResults(:,1) = midx';
        posT = length(find(trainResults(:,1) - trainResults(:,2)==0));
        fprintf('result %d/%d (%3.2f%%), ellapsed time %4.4f\n',...
            posT,samples,posT/samples*100,toc);

        %% test set
        fprintf(' test set:\t'); tic;
        output = fpropBatch(testMat,batchSize,fprops,expand,weiComp,prms,auxPrms,T);
        [~,midx] = max(output(:,:,T));
        testResults(:,1) = midx';
        posT = length(find(testResults(:,1) - testResults(:,2)==0));
        fprintf('result %d/%d (%3.2f%%), ellapsed time %4.4f\n',...
            posT,testSamples,posT/testSamples*100,toc);

        %% voting
        posT = votingSummary(patch,testResults,dims{end});
        
        %% save parameters
        if preResult < posT
            preResult = posT;
            save(strcat(home,'/trained/',name,'/',name,'_',timeStr,'.mat'),'fprops','prms','auxPrms');
        end        

        %% gradient accuracy
        for l=gcloop:L
            figure(3)
            rebuf = squeeze(rerror{l}(:,3,:))';
            subplot(L,2,l*2-1);
            plot(rebuf);
            subplot(L,2,l*2);
            plot(log10(rebuf));ylim([-9 -3]);
            drawnow

            [r,c,v] = ind2sub(size(rerror{l}),find(rerror{l}>tol));
            idx = find(c==3);
            if isempty(idx)~=1, fprintf(' relative error > %e\n',tol); end
            for i=1:length(idx)
                disp([r(idx(i)) rerror{l}(r(idx(i)),:,v(idx(i))) v(idx(i))]);
            end
        end

        %% end of main loop
        fprintf('\n');
    end

    %% report the best result
    fid = fopen(strcat(home,'/logs/',name,'/score report'),'a');
    fprintf(fid,'%s %s %d\n',name,timeStr,preResult);
    fclose(fid);
    
    %% stop taking log
    diary off
end