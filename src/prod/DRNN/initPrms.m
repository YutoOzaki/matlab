function prms = initPrms(vis,hid)
    W = 2.*(rand(hid,vis) - 0.5) .* sqrt(6/(vis+hid));
    b = 2.*(rand(hid,1) - 0.5) .* sqrt(6/hid);
    
    prms = cell(1,2);
    prms{1,1} = W;
    prms{1,2} = b;
end