function [mu_c] = calcular_muc(r,x,m_c)
  mu_c = zeros(1,length(ric(1,:)))
  for c = 1:length(ric(1,:))
    mu_c(c) = sum(sum(r*x))/m_c
  endfor
endfunction