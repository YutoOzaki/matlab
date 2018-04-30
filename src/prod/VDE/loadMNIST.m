function loadMNIST
    load('..\..\data\MNIST\MNISTdataset.mat');
    data = [traindata testdata];
    %data = traindata;
    save mnistdata data;
end