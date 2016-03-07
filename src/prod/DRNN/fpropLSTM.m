function [output,states] = fpropLSTM(input,prms,T,states)
    
    W_z = prms{1,1};
    W_f = prms{1,2};
    W_i = prms{1,3};
    W_o = prms{1,4};
    R_z = prms{1,5};
    R_f = prms{1,6};
    R_i = prms{1,7};
    R_o = prms{1,8};
    P_f = prms{1,9};
    P_i = prms{1,10};
    P_o = prms{1,11};
    b_z = prms{1,12};
    b_f = prms{1,13};
    b_i = prms{1,14};
    b_o = prms{1,15};
    
    z = states{1,1};
    h = states{1,2};
    c = states{1,3};
    F = states{1,4};
    I = states{1,5};
    O = states{1,6};
    gF = states{1,7};
    gI = states{1,8};
    gO = states{1,9};
    u = states{1,10};
    
    batchSize = size(input,2);

    b_zMat = repmat(b_z,1,batchSize);
    b_fMat = repmat(b_f,1,batchSize);
    b_iMat = repmat(b_i,1,batchSize);
    b_oMat = repmat(b_o,1,batchSize);
    P_fMat = repmat(P_f,1,batchSize);
    P_iMat = repmat(P_i,1,batchSize);
    P_oMat = repmat(P_o,1,batchSize);
        
    for t=1:T
        x_t = input(:,:,t);
        u_t = u(:,:,t);
        c_t = c(:,:,t);

        z(:,:,t) = W_z*x_t + R_z*u_t + b_zMat;
        h(:,:,t) = tanh(z(:,:,t));

        F(:,:,t) = W_f*x_t + R_f*u_t + b_fMat + P_fMat.*c_t;
        gF(:,:,t) = sigmoid(F(:,:,t));
        I(:,:,t) = W_i*x_t + R_i*u_t + b_iMat + P_iMat.*c_t;
        gI(:,:,t) = sigmoid(I(:,:,t));

        c_t = h(:,:,t).*gI(:,:,t) + c_t.*gF(:,:,t);
        O(:,:,t) = W_o*x_t + R_o*u_t + b_oMat + P_oMat.*c_t;
        gO(:,:,t) = sigmoid(O(:,:,t));
        u_t = tanh(c_t).*gO(:,:,t);

        c(:,:,t+1) = c_t;
        u(:,:,t+1) = u_t;
    end
    
    states{1,1} = z;
    states{1,2} = h;
    states{1,3} = c;
    states{1,4} = F;
    states{1,5} = I;
    states{1,6} = O;
    states{1,7} = gF;
    states{1,8} = gI;
    states{1,9} = gO;
    states{1,10} = u;
    
    output = u(:,:,2:T+1);
end