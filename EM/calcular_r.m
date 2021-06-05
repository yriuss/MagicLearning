function [r] = calcular_r(p)
  for i = 1:length(p)
    r(i) = p[i,:]*normal(x,uc,Sc);
  endfor
  r = r/sum(r);
endfunction