function drawdist(pi, theta, phi, fignum)
    figure(fignum); 
    subplot(1,12,1); imagesc(pi); caxis([0 1]); set(gca, 'XTick', []); title('global-level');
    subplot(1,12,3:12); imagesc(theta); caxis([0 1]); title('local-level');
    
    figure(fignum+1); imagesc(phi'); caxis([0 1]); title('topic-word distribution');
    
    drawnow();
end