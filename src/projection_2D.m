function [sortie] = projection_2D(data,moyenne,i1,i2,u)
    U = [u(:,i1) u(:,i2)]; 
    data_sortie = (data-moyenne)' * U; 
    sortie = data_sortie; 
end

