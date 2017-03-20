function checkInteger(input, varname)
    assert(isequal(ceil(input), floor(input)), sprintf('index array %s should be integer', varname));
end