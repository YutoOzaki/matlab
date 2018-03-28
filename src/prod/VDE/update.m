function update(nets)
    netnames = fieldnames(nets);
    
    for i=1:length(netnames)
        nodenames = fieldnames(nets.(netnames{i}));
        
        for j=1:length(nodenames)
            nets.(netnames{i}).(nodenames{j}).update();
        end
    end
end