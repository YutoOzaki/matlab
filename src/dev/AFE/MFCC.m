function MFCC(x,fs,hprms,label)
    %% get hyperparameters
    mfb_type    = hprms.mfb_type;
    f_start     = hprms.f_start;
    f_end       = hprms.f_end;
    coefnum     = hprms.coefnum;
    coef_range  = hprms.coef_range;
    unitpow     = hprms.unitpow;
    liftering   = hprms.liftering;
    liftcoef    = hprms.liftcoef;
    mfbc        = hprms.mfbc;
    d           = hprms.d;
    dd          = hprms.dd;
    N           = hprms.N;
    M           = hprms.M;
    preemp      = hprms.preemp;
    w           = hprms.w;
    blocks      = hprms.blocks;
    patch       = hprms.patch;
    normalize   = hprms.normalize;
    zcaeps      = hprms.zcaeps;
    
    %% calculate internal parameters
    FFTL = 2^nextpow2(N); %if 618 then 1024
    w = w(N);
    FFTL_half = 1:(floor(FFTL/2) + 1);

    %coef = 40; unitpow;impl;isdct;eps = 1e-5;f_s = 22050;

    if liftering == 1
        ceplifter = 1 + (liftcoef/2)*sin(pi*(0:coef-1)'/liftcoef);
    elseif liftering == 2
        a = liftcoef(1);
        tau = liftcoef(2);
        ceplifter = ((1:coef).^a).*exp(-((1:coef).^2)./(2*tau^2));
    else
        ceplifter = ones(coef,1);
    end

    %% Check sampling rate of audio files
    %{
    for i=1:samples
        fprintf('\nCheck sampling rate (%d Hz)... %d/%d\n', f_s, i, samples);

        audioFile = url{rndidx(i),1};
        [~,check] = audioread(audioFile);

        if check ~= f_s
            fprintf('\nunexpected sampling rate: %d Hz\n%s\n', check, audioFile);
        end
    end
    %}

    %% Filter bank
    if impl == 1
        F = (0:FFTL/2)/FFTL*f_s;
        cbin = zeros(coef+2,1);
        Mel1 = 2595*log10(1+f_start/700);
        Mel2 = 2595*log10(1+(f_end)/700);
        f_c = zeros(coef+2,1);
        f_c(1) = f_start;
        f_c(end) = f_end;
        for i=1:coef
            f_c(i+1) = Mel1 + (Mel2-Mel1)/(coef+1)*i;
            f_c(i+1) = (10^(f_c(i+1)/2595) - 1)*700;
            cbin(i+1) = round(f_c(i+1)/f_s * FFTL) + 1;
        end
        cbin(1) = round(f_start/f_s*FFTL) + 1;
        cbin(coef+2) = round(f_end/f_s*FFTL) + 1;

        trifil = zeros(coef,1+FFTL/2);
        for k=2:coef+1
            for i=cbin(k-1):cbin(k)
                trifil(k-1,i) = (i-cbin(k-1)+1)/(cbin(k)-cbin(k-1)+1);
            end
            for i=cbin(k)+1:cbin(k+1)
                trifil(k-1,i) = (1 - (i-cbin(k))/(cbin(k+1)-cbin(k)+1));
            end
        end

        if unitPow == 1
            filterArea = sum(trifil,2);
            for i=1:coef
                trifil(i,:) = trifil(i,:)./filterArea(i);
            end
        end
    else
        [trifil,f_c] = mfccFB40(FFTL,f_s);
        trifil = trifil(FFTL_half,:)';
    end

    %% get block-wise features
    [testMat,testLabel,testSourceInfo] = blockWiseMFCC(testURL,mapObj,coef_range,blocks,patch,N,M,preemp,w,trifil,ceplifter,isdct);
    [trainMat,trainLabel,trainSourceInfo] = blockWiseMFCC(trainURL,mapObj,coef_range,blocks,patch,N,M,preemp,w,trifil,ceplifter,isdct);

    %{
    bidim = [40 50]; 
    trainMat = blockSummary(trainMat,coef_range,bidim);
    testMat = blockSummary(testMat,coef_range,bidim);
    %}

    %% Data normalization
    mu = []; stdev = []; V = []; D = [];

    if normalize(1) == 1
        mu = mean(trainMat,2);
        trainMat = trainMat - repmat(mu,1,length(trainLabel));
        testMat = testMat - repmat(mu,1,length(testLabel));
    end

    if normalize(2) == 1
        stdev = std(trainMat,1,2);
        trainMat = trainMat./repmat(stdev,1,length(trainLabel)); 
        testMat = testMat./repmat(stdev,1,length(testLabel)); 
    end

    if normalize(3) == 1
        C = cov(trainMat');
        [V,D] = eig(C);
        ZCA = V*diag(diag(D+eps).^(-0.5))*V';

        trainMat = ZCA * trainMat;
        testMat = ZCA * testMat;

        figure(2);
        subplot(311);surf(ZCA,'LineStyle','None','EdgeColor','None');view(0,90);colormap gray
        C_zca = cov(trainMat');
        subplot(312);surf(C_zca,'LineStyle','None','EdgeColor','None');view(0,90);colormap gray
        T_zca = cov(testMat');
        subplot(313);surf(T_zca,'LineStyle','None','EdgeColor','None');view(0,90);colormap gray
    end  

    %% Save result
    save(fileName,'trainMat','trainLabel','testMat','testLabel','trainSourceInfo','testSourceInfo');
    save(strcat(fileName,'_meta'),'mu','stdev','V','D');
end