function [x, vocab] = loadNIPSdata
    datapath = 'C:\Users\yuto\Documents\MATLAB\data\topicmodel\UCI-NIPS\docword.nips.txt';
    vocabpath = 'C:\Users\yuto\Documents\MATLAB\data\topicmodel\UCI-NIPS\vocab.nips.txt';
    
    fileID = fopen(datapath, 'r');
    data = textscan(fileID, '%s');
    fclose(fileID);
    data = data{1};
    
    fileID = fopen(vocabpath, 'r');
    vocab = textscan(fileID, '%s');
    fclose(fileID);
    vocab = vocab{1};
    
    N = str2double(data{1});
    V = length(vocab);
    assert(V == str2double(data{2}), 'Number of words in the vocabulary is inconsistent');
    x = zeros(V, N);
    
    fprintf('formatting data (%d documents)...', N);
    NWZ = str2double(data{3});
    for i=1:NWZ
        idx = 3*i;

        n = str2double(data{idx + 1});
        v = str2double(data{idx + 2});
        count = str2double(data{idx + 3});
        
        x(v, n) = count;
    end
    fprintf('completed\n');
end