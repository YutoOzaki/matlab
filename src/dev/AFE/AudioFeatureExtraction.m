function [dataMat,dataLabel] = AudioFeatureExtraction(dataSet,hprms)    
    %% read hyperparameters
    method = hprms.method;
    
    path = dataSet.dataPath;
    label = dataSet.dataLabel;
    samples = length(path);

    %% implement audio feature extractor
    featureExtractor = cell(1,1);
    if strcmp(method,'MFCC')
        featureExtractor{1} = @MFCC;
    elseif strcmp(method,'CR')
        featureExtractor{1} = @CR;
    end

    %% extract audio features
    for i=1:samples
        pathBuf = strrep(path{i},'\','/');
        pathBuf = strsplit(pathBuf,'/');
        fprintf('class %d %s (%d/%d)\n',label(i),pathBuf{end},i,samples);
        
        [x,fs] = audioread(path{i});
        %featureExtractor{1}(x,fs,hprms,label(i));
    end
end