function [prms,adadMat] = AdaDelta(prms,gprms,misc,adadMat,coefs)    
    rho = coefs(1);
    beta = 1 - rho;
    eps = coefs(2);

    Egt = adadMat{1};
    Edxt = adadMat{2};
    
    L = length(prms);

    for i=1:L
        theta = prms{i};
        gtheta = gprms{i};
        
        Egtheta = Egt{i};
        Edtheta = Edxt{i};
        
        Egtheta = rho.*Egtheta + beta.*gtheta.^2;
        
        RMSgtheta = sqrt(Egtheta + eps);
        RMSdtheta = sqrt(Edtheta + eps);
        
        dtheta = -RMSdtheta./RMSgtheta.*gtheta;
        
        Edtheta = rho.*Edtheta + beta.*dtheta.^2;
        
        theta = theta + dtheta;
        
        Egt{i} = Egtheta;
        Edxt{i} = Edtheta;
        
        prms{i} = theta;
    end
    
    adadMat{1} = Egt;
    adadMat{2} = Edxt;
end