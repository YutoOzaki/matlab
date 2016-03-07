function states = initLSTMState(hid,batchSize,T)
    stateNum = 10;
    
    z = zeros(hid,batchSize,T);
    h = zeros(hid,batchSize,T);

    c = zeros(hid,batchSize,T+1);
    
    F = zeros(hid,batchSize,T);
    I = zeros(hid,batchSize,T);
    O = zeros(hid,batchSize,T);
    
    gF = zeros(hid,batchSize,T+1);
    gI = zeros(hid,batchSize,T);
    gO = zeros(hid,batchSize,T);
    
    u = zeros(hid,batchSize,T+1);
    
    states = cell(1,stateNum);    
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
end