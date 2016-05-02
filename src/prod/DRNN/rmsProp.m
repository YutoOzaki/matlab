function [prms,rms] = rmsProp(prms, gprms, rms, lrate, drate, eps)
    beta = 1 - drate;

    for i=1:length(prms)
        rms{i} = drate.*rms{i} + beta.*(gprms{i}.^2);

        prms{i} = prms{i} - lrate.*gprms{i}./sqrt(rms{i} + eps);
    end
end