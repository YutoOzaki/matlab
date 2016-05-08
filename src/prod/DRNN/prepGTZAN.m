function dataSet = prepGTZAN(batchSize, N)
    gtzanFilePath = 'C:\Users\yuto\Documents\MATLAB\data\gtzan\gtzan.txt';
    N_train = N(1);
    N_valid = N(2);
    N_test = N(3);
    
    pathList = GTZAN.readFilePath(gtzanFilePath);
    [gtzan_train, gtzan_valid, gtzan_test] = GTZAN(pathList, batchSize, 'whole').stratify(N_train, N_valid, N_test);
    dataSet = {gtzan_train, gtzan_valid, gtzan_test};
    
    for i=1:3
        dataSet{i}.setAFE('MFCC');
    end
end

%{
tic;
while dataSet{1}.setNextIndex();
    dataSet{1}.getFeature();
end
toc;

tic;
while dataSet{2}.setNextIndex();
    dataSet{2}.getFeature();
end
toc;

tic;
while dataSet{3}.setNextIndex();
    dataSet{3}.getFeature();
end
toc;
%}