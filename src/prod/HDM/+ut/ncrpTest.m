function tests = ncrpTest
    tests = functiontests(localfunctions);
end

function testInit1Level(testCase)
    eta = 1;
    
    z = zeros(1000, 1);
    
    z = ncrp(eta, z);
    
    commonVerification(testCase, z, eta);
    commonPlot(z);
end

function testInit2Level(testCase)
    eta = [1 1];
    
    z = zeros(1000, 1);
    
    z = ncrp(eta, z);
    
    commonVerification(testCase, z, eta);
    commonPlot(z);
end

function testInitRandLevel(testCase)
    eta = ones(2 + randi(3), 1);
    
    z = zeros(1000, 1);
    
    z = ncrp(eta, z);
    
    commonVerification(testCase, z, eta);
    commonPlot(z);
end

function testResamplingDense(testCase)
    eta = ones(2 + randi(3), 1);
    
    z = zeros(1000, 1);
    
    z = ncrp(eta, z);
    
    eta = 10 .* eta;
    z = ncrp(eta, z);
    
    commonVerification(testCase, z, eta);
    commonPlot(z);
end

function testResamplingSparse(testCase)
    eta = ones(2 + randi(3), 1);
    
    z = zeros(1000, 1);
    
    z = ncrp(eta, z);
    
    eta = 0.01 .* eta;
    z = ncrp(eta, z);
    
    commonVerification(testCase, z, eta);
    commonPlot(z);
end

function setupOnce(testCase)
end

function teardown(testCase)
end

function commonVerification(testCase, z, eta)
    c = size(z, 2);
    verifyEqual(testCase, length(eta), c);
    
    A = sortrows(z, c);
    d = diff(A);
    
    B = [0;1];
    for i=2:c
        verifyEqual(testCase, unique(d(:, i)), B);
    end
    
    verifySBPTreeCase(testCase, z)
end

function verifySBPTreeCase(testCase, z)
    N = size(z, 1);
    z = [ones(N, 1) z (1:N)'];
    L = size(z, 2);
    
    for l=2:L
        J = length(unique(z(:, l)));

        for j=1:J
            idx = find(z(:, l) == j);
            verifyEqual(testCase, z(idx, l - 1), repmat(z(idx(1), l - 1), length(idx), 1));
        end
    end
end

function commonPlot(z)
    c = size(z, 2);
    z = sortrows(z, c);
    d = diff(z);
    
    figure;
    for i=1:c
        subplot(1, c, i); 
        imagesc(z(:, i));
        set(gca, 'XTick', []); title(sprintf('K = %d', max(z(:, i))));
    end
end