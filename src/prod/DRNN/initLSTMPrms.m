function prms = initLSTMPrms(vis,hid)
    lstmPrmNum = 15;

    W_z = 2.*(rand(hid,vis) - 0.5) .* sqrt(6/(vis+hid));
    W_f = 2.*(rand(hid,vis) - 0.5) .* sqrt(6/(vis+hid));
    W_i = 2.*(rand(hid,vis) - 0.5) .* sqrt(6/(vis+hid));
    W_o = 2.*(rand(hid,vis) - 0.5) .* sqrt(6/(vis+hid));

    R_z = 2.*(rand(hid,hid) - 0.5) .* sqrt(6/(hid+hid));
    R_f = 2.*(rand(hid,hid) - 0.5) .* sqrt(6/(hid+hid));
    R_i = 2.*(rand(hid,hid) - 0.5) .* sqrt(6/(hid+hid));
    R_o = 2.*(rand(hid,hid) - 0.5) .* sqrt(6/(hid+hid));

    P_f = 2.*(rand(hid,1) - 0.5) .* sqrt(6/hid);
    P_i = 2.*(rand(hid,1) - 0.5) .* sqrt(6/hid);
    P_o = 2.*(rand(hid,1) - 0.5) .* sqrt(6/hid);

    b_z = 2.*(rand(hid,1) - 0.5) .* sqrt(6/hid);
    b_f = 2.*(rand(hid,1) - 0.5) .* sqrt(6/hid);
    b_i = 2.*(rand(hid,1) - 0.5) .* sqrt(6/hid);
    b_o = 2.*(rand(hid,1) - 0.5) .* sqrt(6/hid);
    
    prms = cell(1,lstmPrmNum);
    prms{1,1} = W_z;
    prms{1,2} = W_f;
    prms{1,3} = W_i;
    prms{1,4} = W_o;
    prms{1,5} = R_z;
    prms{1,6} = R_f;
    prms{1,7} = R_i;
    prms{1,8} = R_o;
    prms{1,9} = P_f;
    prms{1,10} = P_i;
    prms{1,11} = P_o;
    prms{1,12} = b_z;
    prms{1,13} = b_f;
    prms{1,14} = b_i;
    prms{1,15} = b_o;
end