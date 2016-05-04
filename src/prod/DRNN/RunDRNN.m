%% assertion for project name
assert(exist('name','var')==1,'specify the project name');

%% check machine
if ispc==1
    fprintf('--Running on Windows machine ');
elseif isunix==1
    fprintf('--Running on Linux or Mac OS machine ');
else
    warning('--This machine is neither Windows, Linux, nor Mac OS ');
end

gpuInfo = gpuDevice;
if gpuInfo.DeviceSupported && strcmp(gpuInfo.ComputeMode,'Prohibited') == false && false
    fprintf('(with GPU)--\n');
    drnn = @DRNN_GPU;
else
    fprintf('(with CPU)--\n');
    drnn = @DRNN;
end

%% get hyperparameters
hprms = modeler();

%% run deep recurrent neural network
for i=1:length(hprms)
    drnn(name,hprms(i));
end