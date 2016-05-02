function [prms, moments] = adam(prms, gprms, moments, alpha, drate1, drate2, eps)
    m = moments{1};
    v = moments{2};
    t = moments{3};
    
    beta1 = 1 - drate1;
    beta2 = 1 - drate2;

    for i=1:length(prms)
        m{i} = drate1.*m{i} + beta1.*gprms{i};
        v{i} = drate2.*v{i} + beta2.*gprms{i}.^2;
        
        mt = m{i}./(1 - drate1^t);
        vt = v{i}./(1 - drate2^t);
        
        prms{i} = prms{i} - alpha.*mt./sqrt(vt + eps);
    end
    
    moments{1} = m;
    moments{2} = v;
    moments{3} = t + 1;
end