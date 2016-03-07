function [delta,gprms] = bpropSoftmax(d,input,prms,T,misc)
    W = prms{1,1};

    vis = size(W,2);
    batchSize = size(d,2);
    delta = zeros(vis,batchSize,T);
    
    gprms = cell(1,2);
    gradW = 0;
    gradb = 0;

    for t=T:-1:1
        gradW = gradW + d(:,:,t) * input(:,:,t)';
        gradb = gradb + d(:,:,t);
        
        delta(:,:,t) = W'*d(:,:,t);
    end
    
    gprms{1,1} = gradW./batchSize;
    gprms{1,2} = mean(gradb,2);
end