function [dataMat,dataLabel] = AudioFeatureExtraction(dataSet,hprms)    
    %% read hyperparameters
    method      = hprms.method;
    d           = hprms.d;
    dd          = hprms.d;
    dim         = length(hprms.coef_range);
    blocks      = hprms.blocks;
    patch       = hprms.patch;
    checkfs     = hprms.checkfs;
    
    hprms.w     = hprms.w(hprms.N);
    
    
    path        = dataSet.dataPath;
    label       = dataSet.dataLabel;
    samples     = length(path);

    %% implement audio feature extractor
    featureExtractor = cell(1,1);
    if strcmp(method,'MFCC')
        featureExtractor{1} = @MFCC;
    elseif strcmp(method,'CR')
        featureExtractor{1} = @CR;
    end
    
    %% prepare variables
    if d == true && dd == false
        dim = dim * 2;
    elseif d == true && d == true
        dim = dim * 3;
    end
    
    dataMat = zeros(dim,samples*patch,blocks);
    dataLabel = zeros(1,samples*patch,blocks);

    %% extract audio features
    for i=1:samples
        pathBuf = strrep(path{i},'\','/');
        pathBuf = strsplit(pathBuf,'/');
        fprintf('class %d %s (%d/%d)\n',label(i),pathBuf{end},i,samples);
        
        [x,fs] = audioread(path{i});
        if fs ~= checkfs
            warning(' Unexpected sampling frequency (%d)',fs);
        end
        %featureExtractor{1}(x,fs,hprms,label(i));
    end
end