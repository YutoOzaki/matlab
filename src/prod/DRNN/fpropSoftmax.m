function [y,misc] = fpropSoftmax(input,prms,T,misc)
    W = prms{1,1};
    b = prms{1,2};
    
    hid = size(W,1);
    batchSize = size(input,2);
    
    y = zeros(hid,batchSize,T);   
    bMat = repmat(b,1,batchSize);

    for t=1:T
        u_t = input(:,:,t);
        
        v = W*u_t + bMat;
        K = max(v);
        v = v - repmat(K,hid,1);
        y(:,:,t) = exp(v)./repmat(sum(exp(v)),hid,1);
    end
end