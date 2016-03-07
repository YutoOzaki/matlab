function [prms,rmsPrms] = rmsProp(prms,gprms,lrate,rmsPrms,coefs)
    drate = coefs(1);
    beta = 1 - drate;
    eps = coefs(2);

    rmsPrms = updateRMSP(gprms,rmsPrms,drate,beta);

    L = length(rmsPrms);

    for i=1:L
        theta = prms{i};
        rms_theta = rmsPrms{i};
        gtheta = gprms{i};
        
        dtheta = -(lrate./sqrt(rms_theta + eps)).*gtheta;
        prms{i} = theta + dtheta;
    end
end