function rmsPrms = updateRMSP(gprms,rmsPrms,drate,beta)
    l = length(rmsPrms);
    
    for i=1:l
        rp_theta = rmsPrms{i};
        gtheta = gprms{i};
        
        rp_theta = drate.*rp_theta + beta.*(gtheta.^2);
        rmsPrms{i} = rp_theta;
    end
end