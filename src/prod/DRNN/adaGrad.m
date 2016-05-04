function [prms, grads] = adaGrad(prms, gprms, grads, lrate)
    for i=1:length(prms)
        grads{i} = grads{i} + gprms{i}.^2;

        prms{i} = prms{i} - lrate.*gprms{i}./sqrt(grads{i});
    end
end