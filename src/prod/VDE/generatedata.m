function generatedata
    N = 4000;
    D = 3;
    numclass = 5;
    data = testdata(N, D, numclass);
    
    save('testdata.mat', 'data');
end