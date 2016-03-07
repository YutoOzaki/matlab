function [delta,gprms] = bpropBLSTM(d,input,prms,T,states)    
    dsigmoid = @(x) sigmoid(x).*(1 - sigmoid(x));
    dtanh = @(x) 1 - tanh(x).^2;
    gprms = cell(1,30);
    
    d_f = d{1};
    d_b = d{2};
    delta = cell(1,2);
    
    %% forward direction
    gradW_z = 0; gradR_z = 0; gradb_z = 0;
    gradW_o = 0; gradR_o = 0; gradb_o = 0; gradP_o = 0;
    gradW_f = 0; gradR_f = 0; gradb_f = 0; gradP_f = 0;
    gradW_i = 0; gradR_i = 0; gradb_i = 0; gradP_i = 0;
    
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
    
    dz = z(:,:,1).*0;
    dI = I(:,:,1).*0;
    dF = F(:,:,1).*0;
    dO = O(:,:,1).*0;
    dc = c(:,:,1).*0;
    
    batchSize = size(input{1},2);
    vis = size(W_z,2);
    delta_f = zeros(vis,batchSize,T);
    
    P_fMat = repmat(P_f,1,batchSize);
    P_iMat = repmat(P_i,1,batchSize);
    P_oMat = repmat(P_o,1,batchSize);
    
    for t=T:-1:1
        du = d_f(:,:,t) + R_z'*dz + R_i'*dI + R_f'*dF + R_o'*dO;
        dO = du.*tanh(c(:,:,t+1)).*dsigmoid(O(:,:,t));
        dc = du.*gO(:,:,t).*dtanh(c(:,:,t+1)) + P_oMat.*dO + P_iMat.*dI...
            + P_fMat.*dF + dc.*gF(:,:,t+1);
        dF = dc.*c(:,:,t).*dsigmoid(F(:,:,t));
        dI = dc.*h(:,:,t).*dsigmoid(I(:,:,t));
        dz = dc.*gI(:,:,t).*dtanh(z(:,:,t));
        
        delta_f(:,:,t) = W_z'*dz + W_i'*dI + W_f'*dF + W_o'*dO;

        gradR_z = gradR_z + dz*u(:,:,t)';
        gradR_f = gradR_f + dF*u(:,:,t)';
        gradR_i = gradR_i + dI*u(:,:,t)';
        gradR_o = gradR_o + dO*u(:,:,t)';

        gradP_f = gradP_f + dF.*c(:,:,t);
        gradP_i = gradP_i + dI.*c(:,:,t);
        gradP_o = gradP_o + dO.*c(:,:,t+1);

        gradW_z = gradW_z + dz*input{1}(:,:,t)';
        gradW_f = gradW_f + dF*input{1}(:,:,t)';
        gradW_i = gradW_i + dI*input{1}(:,:,t)';
        gradW_o = gradW_o + dO*input{1}(:,:,t)';

        gradb_z = gradb_z + dz;
        gradb_f = gradb_f + dF;
        gradb_i = gradb_i + dI;
        gradb_o = gradb_o + dO;
    end
    
    gprms{1,1} = gradW_z./batchSize;
    gprms{1,2} = gradW_f./batchSize;
    gprms{1,3} = gradW_i./batchSize;
    gprms{1,4} = gradW_o./batchSize;
    
    gprms{1,5} = gradR_z./batchSize;
    gprms{1,6} = gradR_f./batchSize;
    gprms{1,7} = gradR_i./batchSize;
    gprms{1,8} = gradR_o./batchSize;
    
    gprms{1,9} = mean(gradP_f,2);
    gprms{1,10} = mean(gradP_i,2);
    gprms{1,11} = mean(gradP_o,2);
    
    gprms{1,12} = mean(gradb_z,2);
    gprms{1,13} = mean(gradb_f,2);
    gprms{1,14} = mean(gradb_i,2);
    gprms{1,15} = mean(gradb_o,2);
    
    %% backward direction
    gradW_z = 0; gradR_z = 0; gradb_z = 0;
    gradW_o = 0; gradR_o = 0; gradb_o = 0; gradP_o = 0;
    gradW_f = 0; gradR_f = 0; gradb_f = 0; gradP_f = 0;
    gradW_i = 0; gradR_i = 0; gradb_i = 0; gradP_i = 0;
    
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
    
    dz = z(:,:,1).*0;
    dI = I(:,:,1).*0;
    dF = F(:,:,1).*0;
    dO = O(:,:,1).*0;
    dc = c(:,:,1).*0;
    
    vis = size(W_z,2);
    delta_b = zeros(vis,batchSize,T);
    
    P_fMat = repmat(P_f,1,batchSize);
    P_iMat = repmat(P_i,1,batchSize);
    P_oMat = repmat(P_o,1,batchSize);
    
    for t=1:T
        du = d_b(:,:,t) + R_z'*dz + R_i'*dI + R_f'*dF + R_o'*dO;
        dO = du.*tanh(c(:,:,t)).*dsigmoid(O(:,:,t));
        dc = du.*gO(:,:,t).*dtanh(c(:,:,t)) + P_oMat.*dO + P_iMat.*dI...
            + P_fMat.*dF + dc.*gF(:,:,t);
        dF = dc.*c(:,:,t+1).*dsigmoid(F(:,:,t));
        dI = dc.*h(:,:,t).*dsigmoid(I(:,:,t));
        dz = dc.*gI(:,:,t).*dtanh(z(:,:,t));
        
        delta_b(:,:,t) = W_z'*dz + W_i'*dI + W_f'*dF + W_o'*dO;

        gradR_z = gradR_z + dz*u(:,:,t+1)';
        gradR_f = gradR_f + dF*u(:,:,t+1)';
        gradR_i = gradR_i + dI*u(:,:,t+1)';
        gradR_o = gradR_o + dO*u(:,:,t+1)';

        gradP_f = gradP_f + dF.*c(:,:,t+1);
        gradP_i = gradP_i + dI.*c(:,:,t+1);
        gradP_o = gradP_o + dO.*c(:,:,t);

        gradW_z = gradW_z + dz*input{2}(:,:,t)';
        gradW_f = gradW_f + dF*input{2}(:,:,t)';
        gradW_i = gradW_i + dI*input{2}(:,:,t)';
        gradW_o = gradW_o + dO*input{2}(:,:,t)';

        gradb_z = gradb_z + dz;
        gradb_f = gradb_f + dF;
        gradb_i = gradb_i + dI;
        gradb_o = gradb_o + dO;
    end
    
    gprms{1,16} = gradW_z./batchSize;
    gprms{1,17} = gradW_f./batchSize;
    gprms{1,18} = gradW_i./batchSize;
    gprms{1,19} = gradW_o./batchSize;
    
    gprms{1,20} = gradR_z./batchSize;
    gprms{1,21} = gradR_f./batchSize;
    gprms{1,22} = gradR_i./batchSize;
    gprms{1,23} = gradR_o./batchSize;
    
    gprms{1,24} = mean(gradP_f,2);
    gprms{1,25} = mean(gradP_i,2);
    gprms{1,26} = mean(gradP_o,2);
    
    gprms{1,27} = mean(gradb_z,2);
    gprms{1,28} = mean(gradb_f,2);
    gprms{1,29} = mean(gradb_i,2);
    gprms{1,30} = mean(gradb_o,2);
    
    delta{1} = delta_f;
    delta{2} = delta_b;
end