function [name,trainMat,trainLabel,testMat,testLabel] = trajectoryData(samples,testSamples,T)
    name = 'trajectory';
    dim = 2;
    class = 3;
    
    trainMat = zeros(dim,samples,T);
    trainLabel = zeros(1,samples);
    
    testMat = zeros(dim,testSamples,T);
    testLabel = zeros(1,testSamples);

    dt = 2*pi/30;
    a = 0.7;

    for i=1:samples
        b = 2*pi.*rand;
        t0 = rand*2*pi;
        t = t0:dt:t0+(T-1)*dt;

        l = mod(i,class) + 1;
        trainLabel(i) = l;

        if l==1
            trainMat(:,i,:) = [a.*sin(t+b).*abs(sin(t)); a.*cos(t+b).*abs(sin(t))];
        elseif l==2
            trainMat(:,i,:) = [a.*sin(0.5.*t+b).*sin(1.5.*t); a.*cos(t+b).*sin(2.*t)];
        elseif l==3
            trainMat(:,i,:) = [a.*sin(t+b).*sin(2.*t); a.*cos(t+b).*sin(2.*t)];
        end
    end
    
    for i=1:testSamples
        b = 2*pi.*rand;
        t0 = rand*2*pi;
        t = t0:dt:t0+(T-1)*dt;

        l = mod(i,class) + 1;
        testLabel(i) = l;

        if l==1
            testMat(:,i,:) = [a.*sin(t+b).*abs(sin(t)); a.*cos(t+b).*abs(sin(t))];
        elseif l==2
            testMat(:,i,:) = [a.*sin(0.5.*t+b).*sin(1.5.*t); a.*cos(t+b).*sin(2.*t)];
        elseif l==3
            testMat(:,i,:) = [a.*sin(t+b).*sin(2.*t); a.*cos(t+b).*sin(2.*t)];
        end
    end

    %{
    figure(1);
    startIdx = randi(samples-9);
    for i=1:9
        x(:,:) = trainMat(:,startIdx+i,:);
        subplot(3,3,i);plot(x(1,:),x(2,:));
        xlim([-a a]);ylim([-a a]);
    end
    drawnow;
    %}
end