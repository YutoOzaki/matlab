function [x, vocab] = loadNYTdata
    datapath = 'C:\Users\yuto\Documents\MATLAB\data\topicmodel\NewYorkTimesNews\nyt_data.txt';
    vocabpath = 'C:\Users\yuto\Documents\MATLAB\data\topicmodel\NewYorkTimesNews\nyt_vocab.dat';
    
    fileID = fopen(datapath, 'r');
    data = textscan(fileID, '%s');
    fclose(fileID);
    data = data{1};
    
    fileID = fopen(vocabpath, 'r');
    vocab = textscan(fileID, '%s');
    fclose(fileID);
    vocab = vocab{1};
    
    N = length(data);
    V = length(vocab);
    x = zeros(V, N);
    fprintf('formatting data (%d documents)...', N);
    
    parfor n=1:N
        str = strsplit(data{n}, ',');
        A = cell2mat(cellfun(@(s) str2double(strsplit(s, ':'))', str, 'UniformOutput', false))';
        
        assert(size(A, 1) == length(unique(A(:, 1))), 'bag of words with duplicated elements!');
        
        tmp = zeros(V, 1);
        tmp(A(:, 1)) = A(:, 2);
        
        x(:, n) = tmp;
    end
    
    fprintf('completed\n');
end