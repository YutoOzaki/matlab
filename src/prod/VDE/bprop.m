function bprop(x, encnet, decnet, priornet, lossnode)
    delta = lossnode.backwardprop([]);
    %lossnode.localgradcheck();
    
    delta = priornet.weight.backwardprop(delta);
end