function [n] = normal(x,uc,Sc)
  n = zeros(1,length(x))
  for i = 1:length(x)
      n(i) = (1/((2*pi)^(0.5)*Sc))*exp(-(x(i)-uc)^2/(2*Sc^2));
  end;
endfunction
