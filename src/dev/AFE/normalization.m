function normalization()
    %% Data normalization
    mu = []; stdev = [];

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
end