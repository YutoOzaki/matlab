function output = fpropBatch(dataMat,batchSize,fprops,expand,weiComp,prms,auxPrms,T)
    samples = size(dataMat,2);
    L = length(prms);
    output = zeros(size(prms{L}{1},1),samples,T);

    for m=1:batchSize:samples
        idx = m:m+batchSize-1;
        
        input = dataMat(:,idx,:);
        input = expand(input);
        for l=1:L
            input = weiComp{l}(input);
            input = fprops{l}(input,prms{l},T,auxPrms{l});
        end
        
        output(:,idx,:) = input;
    end
end