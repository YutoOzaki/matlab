function [trainData,validData,testData] = gtzanPrep(datadir)
    %% get path of audio data
    fileID = fopen(strcat(datadir,'gtzan.txt'));
    path = textscan(fileID, '%s');
    fclose(fileID);
    
    path = path{1,1};
    samples = length(path);

    %% define the size of each data category
    trainSize = 900;
    validSize = 0;
    testSize = 100;
    
    assert(samples == trainSize + validSize + testSize,...
        'total size of data is not consistent');
    
    %% define hashmap for mapping genre and label
    keySet =   {'blues', 'classical', 'country', 'disco','hiphop','jazz','metal','pop','reggae','rock'};
    valueSet = [1 2 3 4 5 6 7 8 9 10];
    mapObj = containers.Map(keySet,valueSet);

    %% separate into training, validation and test set
    classNum = 10;
    perClass = 100;
    
    idx = 1:samples;
    testIdx = zeros(1,testSize);
    perSet = testSize/classNum;
    
    for i=1:classNum
        testIdx((i-1)*perSet+1:i*perSet) = idx(randperm(perClass,perSet) + (i-1)*perClass);
    end
    
    perClass = perClass - perSet;
    idx = setdiff(idx,testIdx);
    trainIdx = zeros(1,trainSize);
    perSet = trainSize/classNum;
    
    for i=1:classNum
        trainIdx((i-1)*perSet+1:i*perSet) = idx(randperm(perClass,perSet) + (i-1)*perClass);
    end
    
    perClass = perClass - perSet;
    idx = setdiff(idx,trainIdx);
    validIdx = zeros(1,validSize);
    perSet = validSize/classNum;
    
    for i=1:classNum
        validIdx((i-1)*perSet+1:i*perSet) = idx(randperm(perClass,perSet) + (i-1)*perClass);
    end
    
    assert(isempty(setdiff([testIdx trainIdx validIdx],1:samples)),...
        'separation of data is not agreed');   
    
    trainPath = path(trainIdx);
    validPath = path(validIdx);
    testPath = path(testIdx);
    
   %% create class label array
   trainLabel = zeros(1,trainSize);
   validLabel = zeros(1,validSize);
   testLabel = zeros(1,testSize);
   
    for i=1:trainSize
        pathBuf = strrep(trainPath{i},'\','/');
        label_str = strsplit(pathBuf,'/');
        trainLabel(i) = mapObj(label_str{end-1});
    end
    
    for i=1:validSize
        pathBuf = strrep(validPath{i},'\','/');
        label_str = strsplit(pathBuf,'/');
        validLabel(i) = mapObj(label_str{end-1});
    end
    
    for i=1:testSize
        pathBuf = strrep(testPath{i},'\','/');
        label_str = strsplit(pathBuf,'/');
        testLabel(i) = mapObj(label_str{end-1});
    end
    
    %% data and label
    trainData = struct(...
    'dataPath',{trainPath},...
    'dataLabel',trainLabel...
    );

    validData = struct(...
    'dataPath',{validPath},...
    'dataLabel',validLabel...
    );
    
    testData = struct(...
    'dataPath',{testPath},...
    'dataLabel',testLabel...
    );
end