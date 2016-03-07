function [output,states] = fpropBLSTM(input,prms,T,states)
    input_f = input{1};
    input_b = input{2};
    output = cell(1,2);
    
    batchSize = size(input_f,2);
    
    %% forward direction
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
    
    b_zMat = repmat(b_z,1,batchSize);
    b_fMat = repmat(b_f,1,batchSize);
    b_iMat = repmat(b_i,1,batchSize);
    b_oMat = repmat(b_o,1,batchSize);
    P_fMat = repmat(P_f,1,batchSize);
    P_iMat = repmat(P_i,1,batchSize);
    P_oMat = repmat(P_o,1,batchSize);
        
    for t=1:T
        x_t = input_f(:,:,t);
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
    
    output{1} = u(:,:,2:T+1);
    
    %% backward direction
    W_z = prms{1,16};
    W_f = prms{1,17};
    W_i = prms{1,18};
    W_o = prms{1,19};
    R_z = prms{1,20};
    R_f = prms{1,21};
    R_i = prms{1,22};
    R_o = prms{1,23};
    P_f = prms{1,24};
    P_i = prms{1,25};
    P_o = prms{1,26};
    b_z = prms{1,27};
    b_f = prms{1,28};
    b_i = prms{1,29};
    b_o = prms{1,30};
    
    z = states{1,11};
    h = states{1,12};
    c = states{1,13};
    F = states{1,14};
    I = states{1,15};
    O = states{1,16};
    gF = states{1,17};
    gI = states{1,18};
    gO = states{1,19};
    u = states{1,20};
    
    b_zMat = repmat(b_z,1,batchSize);
    b_fMat = repmat(b_f,1,batchSize);
    b_iMat = repmat(b_i,1,batchSize);
    b_oMat = repmat(b_o,1,batchSize);
    P_fMat = repmat(P_f,1,batchSize);
    P_iMat = repmat(P_i,1,batchSize);
    P_oMat = repmat(P_o,1,batchSize);
        
    for t=T:-1:1
        x_t = input_b(:,:,t);
        u_t = u(:,:,t+1);
        c_t = c(:,:,t+1);

        z(:,:,t) = W_z*x_t + R_z*u_t + b_zMat;
        h(:,:,t) = tanh(z(:,:,t));

        F(:,:,t) = W_f*x_t + R_f*u_t + b_fMat + P_fMat.*c_t;
        gF(:,:,t+1) = sigmoid(F(:,:,t));
        I(:,:,t) = W_i*x_t + R_i*u_t + b_iMat + P_iMat.*c_t;
        gI(:,:,t) = sigmoid(I(:,:,t));

        c_t = h(:,:,t).*gI(:,:,t) + c_t.*gF(:,:,t+1);
        O(:,:,t) = W_o*x_t + R_o*u_t + b_oMat + P_oMat.*c_t;
        gO(:,:,t) = sigmoid(O(:,:,t));
        u_t = tanh(c_t).*gO(:,:,t);

        c(:,:,t) = c_t;
        u(:,:,t) = u_t;
    end
    
    states{1,11} = z;
    states{1,12} = h;
    states{1,13} = c;
    states{1,14} = F;
    states{1,15} = I;
    states{1,16} = O;
    states{1,17} = gF;
    states{1,18} = gI;
    states{1,19} = gO;
    states{1,20} = u;
    
    output{2} = u(:,:,1:T);
end