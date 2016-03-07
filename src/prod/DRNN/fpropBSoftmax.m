function [y,misc] = fpropBSoftmax(input,prms,T,misc)
    input_f = input{1};
    input_b = input{2};
    
    W1 = prms{1,1};
    b = prms{1,2};
    W2 = prms{1,3};
    
    hid = size(W1,1);
    batchSize = size(input_f,2);
    
    y = zeros(hid,batchSize,T);   
    bMat = repmat(b,1,batchSize);

    for t=1:T
        u_tf = input_f(:,:,t);
        u_tb = input_b(:,:,t);
        
        v = W1*u_tf + W2*u_tb + bMat;
        K = max(v);
        v = v - repmat(K,hid,1);
        y(:,:,t) = exp(v)./repmat(sum(exp(v)),hid,1);
    end
end