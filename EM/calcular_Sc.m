function [Sc] = calcular_Sc(xi,muc,mc)
  Sc = sum(ric*(xi-muc)'*(xi-muc))/mc;
endfunction