function labelMat = oneHotVectorLabel(labels,classNum,T)
    N = length(labels);
    labelMat = zeros(classNum,N,T);
    
    for i=1:N
        labelMat(labels(i),i,:) = 1;
    end
end