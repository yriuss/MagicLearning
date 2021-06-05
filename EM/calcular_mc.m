function [m_c] = calcular_mc(ric)
  m_c = zeros(1,length(ric(1,:)))
  for c = 1:length(ric(1,:))
    m_c(c) = sum(ric[c,:])
  endfor
endfunction