function dataSet = prepTestData(batchSize, dataName)
    % 'C:\Users\yuto\Documents\MATLAB\data\gtzan\gtzanMFCC.mat'
    % testdata
    [trainData, validData, testData] = TestData(batchSize, 'whole').stratify(dataName);
    dataSet = {trainData, validData, testData};
end