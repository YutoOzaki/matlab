function [prms,lrdec] = vanillaSGD(prms,gprms,lrate,lrdec,lrmin)
    lrdec(2) = lrdec(1)*lrdec(2);
    lrate = lrate*lrdec(2);

    if lrmin > lrate, lrate = lrmin; end
    
    L = length(prms);
    
    for i=1:L
        theta = prms{i};
        gtheta = gprms{i};
        
        dtheta = -lrate.*gtheta;
        theta = theta + dtheta;
        
        prms{i} = theta;
    end
end