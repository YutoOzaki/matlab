function [prms,rms] = rmsProp(prms,gprms,rms,lrate,drate,beta,eps,prmNum)
    for i=1:prmNum
        rms{i} = drate.*rms{i} + beta.*(gprms{i}.^2);

        prms{i} = prms{i} - lrate.*gprms{i}./sqrt(rms{i} + eps);
    end
end