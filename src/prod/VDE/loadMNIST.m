function loadMNIST
    load('..\..\data\MNIST\MNISTdataset.mat');
    data = [traindata testdata];
    truelabels = [trainlabel testlabel];
    %data = traindata;
    save mnistdata data truelabels;
end