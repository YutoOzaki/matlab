function checkSequence(input, varname)
    N = length(input);
    
    if N > 1
        d = diff(input);
        str = sprintf('index array %s should be sequence', varname);

        if N == 2
            assert(d == 1 || d == -1, str);
        else
            dd = diff(d);

            [r, c] = size(dd);
            zero = zeros(r, c);

            assert(isequal(dd, zero), str);
        end 
    end
end