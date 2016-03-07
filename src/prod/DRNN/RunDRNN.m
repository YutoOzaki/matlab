%% assertion for project name
assert(exist('name','var')==1,'specify the project name');

%% get hyperparameters
hprms = modeler();

%% run deep recurrent neural network
for i=1:length(hprms)
    DRNN(name,hprms(i));
end