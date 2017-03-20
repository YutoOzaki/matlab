function tests = hsbpTest
    tests = functiontests(localfunctions);
end

function testSmallRand(testCase)
    alpha = 1;
    
    x = rand(10, 1);
    x = x./sum(x);
    
    actSolution = hsbp(alpha, x);
    
    commonVerification(actSolution, testCase);
    
    commonPlot(x, actSolution);
end

function testLargeRandQuality(testCase)
    alpha = 1;
    
    for i=1:1000
        x = rand(10000, 1);
        x = x./sum(x);
        
        actSolution = hsbp(alpha, x);

        commonVerification(actSolution, testCase);
    end
    
    commonPlot(x, actSolution);
end

function testDenseDist(testCase)
    alpha = 20;
    
    x = rand(100, 1);
    x = x./sum(x);
    
    actSolution = hsbp(alpha, x);
    
    commonVerification(actSolution, testCase);
    
    commonPlot(x, actSolution);
end

function testSparseDist(testCase)
    alpha = 0.01;
    
    x = rand(100, 1);
    x = x./sum(x);
    
    actSolution = hsbp(alpha, x);
    
    commonVerification(actSolution, testCase);
    
    commonPlot(x, actSolution);
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

function commonPlot(x, actSolution)
    figure;
    
    subplot(211);
    stem(x, 'Marker', 'None'); hold on
    stem(actSolution, '-.m', 'Marker', 'None'); hold off;
    title('Stick breaking weights');
    
    fig = gcf;
    ax = fig.CurrentAxes;
    ax.XLim = [-1 length(x)+1];
    ax.XTick = [];
    
    subplot(212);
    [y1, h1] = getNormRndBase(x);
    stem(h1, y1, 'Marker', 'None'); hold on
    [y2, h2] = getNormRndBase(actSolution);
    stem(h2, y2, '-.m', 'Marker', 'None'); hold off;
    title('Realization of the base measure');
    
    fig = gcf;
    ax = fig.CurrentAxes;
    ax.XTick = [];
end

function [y, h] = getNormRndBase(x)
    ind = mnrnd(1, repmat(x', 500, 1));
    [~, col] = find(ind);
    
    K = length(x);
    h = normrnd(0, 1, [K 1]);
    
    pd = makedist('Normal', 0, 1);
    y = pdf(pd, h(col));
    
    h = h(col);
end