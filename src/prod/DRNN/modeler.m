function hprms = modeler()
    %{
    hprm_man = mansearch();
    hprm_rnd = randsearch();
    hprms = [hprm_man hprm_rnd];
    %}
    hprms = mansearch();
end