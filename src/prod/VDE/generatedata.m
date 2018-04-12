function generatedata
    N = 600;
    D = 2;
    numclass = 4;
    data = testdata(N, D, numclass);
    
    save('testdata.mat', 'data');
end