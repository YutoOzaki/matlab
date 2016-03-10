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
    FFTL = 2^nextpow2(N);
    FFTL_half = 1:(floor(FFTL/2) + 1);

    if liftering == 1
        ceplifter = 1 + (liftcoef/2)*sin(pi*(1:coef)'/liftcoef);
    elseif liftering == 2
        a = liftcoef(1);
        tau = liftcoef(2);
        ceplifter = ((1:coef).^a).*exp(-((1:coef).^2)./(2*tau^2));
    else
        ceplifter = ones(coef,1);
    end

    %% get filter bank
    if mfb_type == 1
        trifil = triangularFB(coefnum,f_start,f_end,FFTL,unitpow);
    else
        trifil = mfccFB40(FFTL,f_s);
        trifil = trifil(FFTL_half,:)';
    end

    %% get block-wise features
    [testMat,testLabel,testSourceInfo] = blockWiseMFCC(testURL,mapObj,coef_range,blocks,patch,N,M,preemp,w,trifil,ceplifter,isdct);
    [trainMat,trainLabel,trainSourceInfo] = blockWiseMFCC(trainURL,mapObj,coef_range,blocks,patch,N,M,preemp,w,trifil,ceplifter,isdct);

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
        ZCA = V*diag(diag(D+zcaeps).^(-0.5))*V';

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