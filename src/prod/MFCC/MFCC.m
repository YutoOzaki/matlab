%% Get url of audio data
home = userpath;
datadir = strcat(home(1:end-1),'/data/gtzan/');
fileID = fopen(strcat(datadir,'gtzan.txt'));clear home
url = textscan(fileID, '%s');
fclose(fileID);

url = url{1,1};
samples = length(url);

%% Parameters
fileName = strcat(datadir,'MFCC_data');

N = 1024; %618?
M = 1024;
FFTL = 2^nextpow2(N); %if 618 then 1024
preemp = [1 0]; %[1 -0.97]
w = hamming(N);
FFTL_half = 1:(floor(FFTL/2) + 1);

f_start = 20;
f_end = 10000; %0.5*fs
coef = 40;
coef_range = 1:40;
unitPow = 1;

liftering = 0;
if liftering > 0
    ceplifter = 1 + (liftering/2)*sin(pi*(0:coef-1)'/liftering);
else
    ceplifter = ones(coef,1);
end
%{
a = 1.5;
tau = 5;
if a > 0, ceplifter = ((1:coef).^a).*exp(-((1:coef).^2)./(2*tau^2)); end
%}

impl = 1;
isdct = 1;

blocks = 60;
patch = 70;

trainSize = 900;
testSize = 100;
classNum = 10;

normalize = [0; 0; 0]; %[mean; standard deviation; ZCA]
eps = 1e-5;

f_s = 22050;

%% Check sampling rate of audio files
%{
for i=1:samples
    fprintf('\nCheck sampling rate (%d Hz)... %d/%d\n', f_s, i, samples);
    
    audioFile = url{rndidx(i),1};
    [~,check] = audioread(audioFile);
    
    if check ~= f_s
        fprintf('\nunexpected sampling rate: %d Hz\n%s\n', check, audioFile);
    end
end
%}

%% Separate into training and test set
testIdx = zeros(1,testSize);
perClass = testSize/classNum;
perSample = samples/classNum;
for i=1:classNum
    testIdx((i-1)*perClass+1:i*perClass) = randperm(perSample,perClass) + perSample*(i-1);
end
trainIdx = setdiff(1:samples,testIdx);

%% Filter bank
if impl == 1
    F = (0:FFTL/2)/FFTL*f_s;
    cbin = zeros(coef+2,1);
    Mel1 = 2595*log10(1+f_start/700);
    Mel2 = 2595*log10(1+(f_end)/700);
    f_c = zeros(coef+2,1);
    f_c(1) = f_start;
    f_c(end) = f_end;
    for i=1:coef
        f_c(i+1) = Mel1 + (Mel2-Mel1)/(coef+1)*i;
        f_c(i+1) = (10^(f_c(i+1)/2595) - 1)*700;
        cbin(i+1) = round(f_c(i+1)/f_s * FFTL) + 1;
    end
    cbin(1) = round(f_start/f_s*FFTL) + 1;
    cbin(coef+2) = round(f_end/f_s*FFTL) + 1;

    trifil = zeros(coef,1+FFTL/2);
    for k=2:coef+1
        for i=cbin(k-1):cbin(k)
            trifil(k-1,i) = (i-cbin(k-1)+1)/(cbin(k)-cbin(k-1)+1);
        end
        for i=cbin(k)+1:cbin(k+1)
            trifil(k-1,i) = (1 - (i-cbin(k))/(cbin(k+1)-cbin(k)+1));
        end
    end

    if unitPow == 1
        filterArea = sum(trifil,2);
        for i=1:coef
            trifil(i,:) = trifil(i,:)./filterArea(i);
        end
    end
else
    [trifil,f_c] = mfccFB40(FFTL,f_s);
    trifil = trifil(FFTL_half,:)';
end

%% Define hashmap for mapping genre and label
keySet =   {'blues', 'classical', 'country', 'disco','hiphop','jazz','metal','pop','reggae','rock'};
valueSet = [0 1 2 3 4 5 6 7 8 9];
mapObj = containers.Map(keySet,valueSet);

%% get block-wise features
trainURL = url(trainIdx);
testURL = url(testIdx);

[testMat,testLabel,testSourceInfo] = blockWiseMFCC(testURL,mapObj,coef_range,blocks,patch,N,M,preemp,w,trifil,ceplifter,isdct);
[trainMat,trainLabel,trainSourceInfo] = blockWiseMFCC(trainURL,mapObj,coef_range,blocks,patch,N,M,preemp,w,trifil,ceplifter,isdct);

%{
bidim = [40 50]; 
trainMat = blockSummary(trainMat,coef_range,bidim);
testMat = blockSummary(testMat,coef_range,bidim);
%}

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
    ZCA = V*diag(diag(D+eps).^(-0.5))*V';

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