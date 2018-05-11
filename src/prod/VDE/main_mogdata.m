function main_mogdata
    K = 10;
    datafile = 'mogdata';
    gpumode = false;
    
    %{
    N = 8000;
    D = 3;
    [data, truelabels] = mogdata(N, D, K);
    save(datafile, 'data', 'truelabels');
    %}
    %load(datafile);
    
    %{
    numepoch = 5;
    h = [128 128 6];
    act = {relutrans() tanhtrans()};
    L = 1;
    pretraining_rpsae(datafile, h, act, L, numepoch, gpumode);
    clf
    %}
    
    %{
    numepoch = 10;
    L = 32;
    fitting_kmeans(datafile, K, L, numepoch, gpumode);
    clf; 
    %}
    
    %{
    numepoch = 5;
    L = 1;
    takeover = true;
    fitting_sgd(datafile, K, L, numepoch, gpumode, takeover);
    clf
    %}
    
    %{
    numepoch = 30;
    L = 1;
    userscript(datafile, L, numepoch, gpumode);
    clf
    %}
    
    load(strcat(datafile, '_vde_clustering_result.mat'));
    hsv = [linspace(0.0, 0.9, K)', 0.8.*ones(K, 1), 0.9.*ones(K, 1)];
    c = hsv2rgb(hsv);
    
    figure(1);
    for k=1:K
        idx = labels == k;
        scatter3(data(1, idx), data(2, idx), data(3, idx), 20, c(k, :));hold on
    end
    hold off;
    
    figure(2);
    for k=1:K
        idx = truelabels == k;
        scatter3(data(1, idx), data(2, idx), data(3, idx), 20, c(k, :));hold on
    end
    hold off;
end