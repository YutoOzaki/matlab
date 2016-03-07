function [trainMat,testMat,trainLabel,testLabel] = prepMFCC(MFCC_data)
    load(MFCC_data);
    
    trainLabel = trainLabel + 1;
    testLabel = testLabel + 1;
    
    trainMat = permute(trainMat,[2 1 3]);
    testMat = permute(testMat,[2 1 3]);
end