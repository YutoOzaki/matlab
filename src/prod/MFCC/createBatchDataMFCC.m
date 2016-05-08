function createBatchDataMFCC()
    gtzanFilePath = 'C:\Users\yuto\Documents\MATLAB\data\gtzan\gtzan.txt';
    batchSize = 10;
    T = 80;
    patch = 120;
    N_train = 900;
    N_valid = 0;
    N_test = 100;
    
    pathList = GTZAN.readFilePath(gtzanFilePath);
    [gtzan_train, gtzan_valid, gtzan_test] = GTZAN(pathList, batchSize, 'whole').stratify(N_train, N_valid, N_test);
    dataSet = {gtzan_train, gtzan_valid, gtzan_test};
    
    for i=1:3
        dataSet{i}.setAFE('MFCC');
    end
    
    dataCell = cell(3,1);
    labelCell = cell(3,1);
    
    for i=1:3
        dataMat = zeros(40,patch*dataSet{i}.N,T);
        dataLabel = zeros(1,patch*dataSet{i}.N);
        j = 0;
        
        while dataSet{i}.setNextIndex()
            batchLog = sprintf('(batch %d - %d)', dataSet{1}.index(1), dataSet{1}.index(end));
            fprintf(batchLog);
            
            mfcc = dataSet{i}.getFeature();
            label = dataSet{i}.getLabel();
            
            for m=1:batchSize
                for k=1:patch
                    t = randi(size(mfcc,3) - T);
                    dataMat(:,j+k,:) = mfcc(:,m,t:t+T-1);
                    dataLabel(:,j+k) = label(m);
                end
                j = j + k;
            end
            
            fprintf(repmat('\b',1,length(batchLog)));
        end
        
        dataCell{i} = dataMat;
        labelCell{i} = dataLabel;
    end
    
    trainMat = dataCell{1};
    trainLabel = labelCell{1};
    
    validMat = dataCell{2};
    validLabel = labelCell{2};
    
    testMat = dataCell{3};
    testLabel = labelCell{3};
    
    save('C:\Users\yuto\Documents\MATLAB\data\gtzan\gtzanMFCC.mat',...
        'trainMat','trainLabel','validMat','validLabel','testMat','testLabel','-v7.3');
end