function prms = initLpPrms(vis,hid)
    W = 2.*(rand(hid(1),vis) - 0.5) .* sqrt(6/(vis+hid(1)));
    c = 2.*(rand(hid(1),1) - 0.5) .* sqrt(6/hid(1));
    
    p = rand(hid(2),1) + 1;
    rho = log(exp(p-1)-1);
    
    prms = cell(1,3);
    prms{1,1} = W;
    prms{1,2} = c;
    prms{1,3} = rho;
end