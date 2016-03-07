function [trainMat,testMat,trainLabel,testLabel] = prepTrajectory()
    samples = 150; testSamples = 30; T = 30;
    [~,trainMat,trainLabel,testMat,testLabel] = trajectoryData(samples,testSamples,T);
end