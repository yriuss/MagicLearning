function [pi_c] = calcular_pic(m_c)
  pi_c = zeros(1,length(m_c(1,:)));
  for c = 1:length(m_c(1,:))
    m_c(c) = m_c(c)/sum(m_c);
  endfor
endfunction