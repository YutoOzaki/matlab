function drawMNIST(x)
    imagesc(reshape(x, [28 28])'); 
    set(gca,'XTick',[]); 
    set(gca,'YTick',[]); 
    colormap gray
end