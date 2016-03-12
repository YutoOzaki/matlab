function normalization()
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
end