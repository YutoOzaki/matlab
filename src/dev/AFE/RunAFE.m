assert(exist('name','var') == 1,'Specify the file name');

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
[trainMat,trainLabel,trainInfo] = AudioFeatureExtraction(trainData,hprms);
[validMat,validLabel,validInfo] = AudioFeatureExtraction(validData,hprms);
[testMat,testLabel,testInfo] = AudioFeatureExtraction(testData,hprms);

%% save parameters
save(strcat(datadir,name),'trainMat','trainLabel','testMat','testLabel','validMat','validLabel');
save(strcat(datadir,name,'_meta'),'trainInfo','validInfo','testInfo');

%% diary off
diary off

%% debugging
%{
N = size(trainMat,2);
xAxis = 1:hprms.blocks;
yAxis = 1:length(hprms.coef_range);
for k=1:10    
    i = randi(N);
    sampleNum = ceil(i/hprms.patch);
    patchNum = (i - (sampleNum-1)*hprms.patch);
    track = trainInfo{sampleNum,1};
    segment = (trainInfo{sampleNum,2}(:,patchNum))./hprms.checkfs;

    bwm = squeeze(trainMat(:,i,:));
    
    subplot(5,2,k);
    surf(xAxis,yAxis,bwm,'LineStyle','none');view(0,90);axis tight;
    title(strcat(track,' (',num2str(segment(1)),' - ',num2str(segment(2)),' sec)'));
end
%}