function newArray = insertElement(array, element, k)
    array_l = array(1:k-1);
    array_u = array(k:end);
    
    newArray = [array_l; element; array_u];
end