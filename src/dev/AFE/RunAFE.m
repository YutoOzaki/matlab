%% enable to collect log
home = userpath;
home = home(1:end-1);
diary off
timeStamp = clock;
timeStr = '';
for i=1:length(timeStamp)-1
    if i~= 1
        timeBuf = num2str(timeStamp(i),'%02i');
    else
        timeBuf = num2str(timeStamp(i));
    end
    timeStr = strcat(timeStr,timeBuf);
end
diary(strcat(home,'/logs/gtzan_MFCC/gtzan_AFE_',timeStr,'.txt'));

%% get hyperparameters
datadir = strcat(home,'/data/gtzan/');
[trainData,validData,testData] = dataPrep(datadir);
hprms = modeler();
disp(hprms);

%% extract audio features from training, validation, and test data
AudioFeatureExtraction(trainData,hprms);
AudioFeatureExtraction(validData,hprms);
AudioFeatureExtraction(testData,hprms);

%% save parameters
%save(fileName,'trainMat','trainLabel','testMat','testLabel','trainSourceInfo','testSourceInfo');
%save(strcat(fileName,'_meta'),'mu','stdev','V','D');

%% diary off
diary off