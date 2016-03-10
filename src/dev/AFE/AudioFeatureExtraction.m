function [dataMat,dataLabel] = AudioFeatureExtraction(dataSet,hprms)    
    %% read hyperparameters
    method      = hprms.method;
    checkfs     = hprms.checkfs;
    
    path        = dataSet.dataPath;
    label       = dataSet.dataLabel;
    samples     = length(path);
    
    featureExtractor = cell(1,1);
    
    %% implement audio feature extractor
    if strcmp(method,'MFCC')
        featureExtractor{1} = @MFCC;
        
        %% matrix for storing audio features
        coefnum     = hprms.coefnum;
        dim         = length(hprms.coef_range);
        d           = hprms.d;
        dd          = hprms.d;
        blocks      = hprms.blocks;
        patch       = hprms.patch;
        
        if d == true && dd == false
            dim = dim * 2;
        elseif d == true && d == true
            dim = dim * 3;
        end
        
        dataMat     = zeros(dim,samples*patch,blocks);
        dataLabel   = zeros(1,samples*patch,blocks);
        
        %% liftering vector
        liftcoef    = hprms.liftcoef;
        liftering   = hprms.liftering;
        if liftering == 1
            hprms.ceplifter = 1 + (liftcoef/2)*sin(pi*(1:coefnum)'/liftcoef);
        elseif liftering == 2
            a = liftcoef(1);
            tau = liftcoef(2);
            hprms.ceplifter = ((1:coefnum).^a).*exp(-((1:coefnum).^2)./(2*tau^2));
        else
            hprms.ceplifter = ones(coefnum,1);
        end
        
        %% mel-filter bank
        mfb_type    = hprms.mfb_type;
        f_start     = hprms.f_start;
        f_end       = hprms.f_end;
        unitpow     = hprms.unitpow;
        FFTL        = 2^nextpow2(hprms.N);
        FFTL_half   = 1:(floor(FFTL/2) + 1);
        if mfb_type == 1
            hprms.melfilbank = triangularFB(coefnum,f_start,f_end,FFTL,unitpow);
        else
            melfilbank = mfccFB40(FFTL);
            hprms.melfilbank = melfilbank(FFTL_half,:)';
        end
    
        %% window function
        hprms.w     = hprms.w(hprms.N);
    elseif strcmp(method,'CR')
        featureExtractor{1} = @CR;
    end
    
    %% extract audio features
    for i=1:samples
        pathBuf = strrep(path{i},'\','/');
        pathBuf = strsplit(pathBuf,'/');
        fprintf('class %d %s (%d/%d)\n',label(i),pathBuf{end},i,samples);
        
        [x,fs] = audioread(path{i});
        if fs ~= checkfs
            warning(' Unexpected sampling frequency (%d)',fs);
        end
        
        idx = (i-1)*patch+1:i*patch;
        dataLabel(1,idx,:) = label(i);
        %dataMat(:,idx,:) = featureExtractor{1}(x,hprms);
    end
end