function [delta,gprms] = bpropBSoftmax(d,input,prms,T,misc)
    W1 = prms{1,1};
    W2 = prms{1,3};

    batchSize = size(d,2);
    delta = cell(1,2);
    
    vis_f = size(W1,2);    
    delta_f = zeros(vis_f,batchSize,T);
    vis_b = size(W2,2);    
    delta_b = zeros(vis_b,batchSize,T);
    
    gprms = cell(1,3);
    gradW1 = 0;
    gradb = 0;
    gradW2 = 0;

    for t=T:-1:1
        gradW1 = gradW1 + d(:,:,t) * input{1}(:,:,t)';
        gradb = gradb + d(:,:,t);
        gradW2 = gradW2 + d(:,:,t) * input{2}(:,:,t)';
        
        delta_f(:,:,t) = W1'*d(:,:,t);
        delta_b(:,:,t) = W2'*d(:,:,t);
    end
    
    gprms{1,1} = gradW1./batchSize;
    gprms{1,2} = mean(gradb,2);
    gprms{1,3} = gradW2./batchSize;
    
    delta{1} = delta_f;
    delta{2} = delta_b;
end