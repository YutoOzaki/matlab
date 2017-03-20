function tests = sbptreeTest
    tests = functiontests(localfunctions);
end

function testUseCase(testCase)
    N = 1000;
    eta = [2 1];
    z = zeros(N, 1);
    z = ncrp(eta, z);

    alpha = [4 4 4 4];
    K = 30;
    
    actSolution = sbptree(alpha, K, z);
    
    L = length(alpha);
    for l=1:L
        J = size(actSolution{l}, 2);
        
        for j=1:J
            commonVerification(actSolution{l}(:, j), testCase);
        end
    end
    
    commonPlot(actSolution, z);
end

function setupOnce(testCase)
end

function teardown(testCase)
end

function commonVerification(actSolution, testCase)
    verifyGreaterThanOrEqual(testCase, actSolution, 0, ...
        @() disp(actSolution(end)));
    
    import matlab.unittest.constraints.HasNaN
    verifyNotEqual(testCase, actSolution, HasNaN, ...
        'The probability vector contains NaN');
    
    verifyEqual(testCase, imag(actSolution), zeros(length(actSolution), 1), 'AbsTol', eps,...
        'The probability vector contains imaginary numbers');
    
    verifyEqual(testCase, sum(actSolution), 1);
end

function commonPlot(actSolution, z)
    N = size(z, 1);
    z = [ones(N, 1) z (1:N)'];
    L = length(actSolution);
    K = length(actSolution{1});
    M = 1024;
    
    pitree = zeros(K, L);
    HtreeY = zeros(M, L);
    HtreeH = zeros(M, L);
    
    figure;
    subplot(1, 2, 1);
    stem(actSolution{1}, 'Marker', 'None');
    title('Stick breaking weights');
    fig = gcf;
    ax = fig.CurrentAxes;
    ax.XLim = [0 K+1];
    ax.XTick = [];
    
    subplot(1, 2, 2);
    [y, h] = getNormRndBase(actSolution{1}, M);
    stem(h, y, 'Marker', 'None');
    title('Realization of the base measure');
    fig = gcf;
    ax = fig.CurrentAxes;
    ax.XLim = [-3 3];
    ax.XTick = [];
    
    idx = find(z(:, 1) == 1);
    I = 1;
    beta = actSolution{1};
    
    pitree(:, 1) = actSolution{I};
    HtreeY(:, 1) = y;
    HtreeH(:, 1) = h;
        
    for l=2:L
        U = unique(z(idx, l));
        J = length(U);
        
        if J > 6
            J = 6;
        end
        
        figure;
        for j=1:J
            subplot(J, 2, 2*j - 1);
            stem(actSolution{l}(:, U(j)), 'Marker', 'None'); hold on
            stem(beta, '-.m', 'Marker', 'None'); hold off
            title(sprintf('Stick breaking weights (parent K = %d)', I));
            fig = gcf;
            ax = fig.CurrentAxes;
            ax.XLim = [0 K+1];
            ax.XTick = [];
            
            subplot(J, 2, 2*j);
            [y, ~] = getNormRndBase(actSolution{l}(:, j), M);
            stem(h, y, 'Marker', 'None');
            title('Realization of the base measure');
            fig = gcf;
            ax = fig.CurrentAxes;
            ax.XLim = [-3 3];
            ax.XTick = [];
        end
        
        I = randi(J) + min(U) - 1;
        idx = find(z(:, l) == I);
        beta = actSolution{l}(:, I);
        
        pitree(:, l) = actSolution{l}(:, I);
        [y, h] = getNormRndBase(actSolution{l}(:, I), M);
        HtreeY(:, l) = y;
        HtreeH(:, l) = h;
    end
    
    figure
    for l=1:L
        subplot(L, 2, 2*l - 1);
        stem(pitree(:, l), 'Marker', 'None');
        fig = gcf;
        ax = fig.CurrentAxes;
        ax.XLim = [0 K+1];
        ax.XTick = [];
        
        subplot(L, 2, 2*l);
        stem(HtreeH(:, l), HtreeY(:, l), 'Marker', 'None');
        fig = gcf;
        ax = fig.CurrentAxes;
        ax.XLim = [-3 3];
        ax.XTick = [];
    end
end

function [y, h] = getNormRndBase(x, M)
    ind = mnrnd(1, repmat(x', M, 1));
    [~, col] = find(ind);
    
    K = length(x);
    h = normrnd(0, 1, [K 1]);
    
    pd = makedist('Normal', 0, 1);
    y = pdf(pd, h(col));
    
    h = h(col);
end