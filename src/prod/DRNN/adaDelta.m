function [prms, exps] = adaDelta(prms, gprms, exps, drate, eps)
    gexp = exps{1};
    xexp = exps{2};
    
    beta = 1 - drate;

    for i=1:length(prms)
        gexp{i} = drate.*gexp{i} + beta.*(gprms{i}.^2);
        
        dx = sqrt(xexp{i}+eps)./sqrt(gexp{i}+eps).*gprms{i};
        
        xexp{i} = drate.*xexp{i} + beta.*(dx.^2);
        
        prms{i} = prms{i} - dx;
    end
    
    exps{1} = gexp;
    exps{2} = xexp;
end