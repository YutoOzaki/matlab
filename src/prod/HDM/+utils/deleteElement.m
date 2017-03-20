function array = deleteElement(array, k)
    if iscell(array)
        L = length(array);

        for l=1:L
            array_tmp = array{l};
            array_tmp(k, :) = [];
            array{l} = array_tmp;
        end
    else
        array(k, :) = [];
    end
end