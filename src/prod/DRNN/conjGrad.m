function [prms,cgprms] = conjGrad(prms,gprms,lrate,cgprms,misc)
    L = length(gprms);
    
    d = cgprms{1};
    beta = cgprms{2};
    pregprms = cgprms{3};
    
    for i=1:L
        theta = prms{i};
        gtheta = gprms{i};       
        dk = d{i};
        bk = beta{i};
        pregtheta = pregprms{i};
        
        dk1 = -gtheta - bk.*dk;
        %/* computation to obtain alpha */
        prms{i} = theta + alpha.*dk1;
        
        bk = -(gtheta'*(gtheta - pregtheta))/(pregtheta'*pregtheta);
        
        d{i} = dk1;
        beta{i} = bk;
    end
    
    pregprms = gprms;
    
    cgprms{1} = d;
    cgprms{2} = beta;
    cgprms{3} = pregprms;
end