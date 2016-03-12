function delta = deltaMFCC(C,i,blocks)
    delta = zeros(size(C,1),blocks);
    bufC = zeros(size(C,1),1);    
    C = [repmat(bufC,1,i) C repmat(bufC,1,i)];
    
    for k=1:blocks
        for t=1:i
            bufC = bufC + i.*(C(:,i+k+t) - C(:,i+k-t));
        end
        
        delta(:,k) = bufC;
        bufC = bufC .* 0;
    end
end