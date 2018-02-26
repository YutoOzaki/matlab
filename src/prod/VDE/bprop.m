function bprop(x, encnet, decnet, priornet, lossnode)
    delta = lossnode.backwardprop([]);
    
    delta = priornet.weight.backwardprop(delta);
    
    %lossnode.localgradcheck();
end