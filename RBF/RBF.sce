function [d] = distance(x1,x2)
    d = sqrt(sum((x1-x2)^2))
endfunction


function [cluster,centroids] = k_means(k, max_iters, X)
    //recebendo a quantidade de dados (row) e a dimensão dos dados (column)
    [row,column] = size(X)
  
  centroids  = []
  
  //gerando os centroides
  
  centroids = [[14.75 1.73 2.39 11.4 91 3.1 3.69 .43 2.81 5.4 1.25 2.73 1150];[12.7 3.87 2.4 23 101 2.83 2.55 .43 1.95 2.57 1.19 3.13 463];...
  [13.73 4.36 2.26 22.5 88 1.28 .47 .52 1.15 6.62 .78 1.75 520]]
  
  
  /*
  //gerar randomicamente
  for i = 1:k,
      centroids = cat(1,centroids,X(grand(1,1,'uin',1,row),:))
  end*/
  
  converged = %f
  current_iter = 0
  
  //inicializando o algorítmo k-means. Ele para se tiver convergido ou o max
  //de iterações for atingido
  while ~converged | current_iter < max_iters,
      distances = zeros(1,length(centroids(:,1)))
      //inicializar clusters
      cluster = zeros(1,length(X(1,:)))
      
      for i = 1:length(X(:,1)),
          
          for c = 1:k,
              distances(c) = distance(X(i,:),centroids(c,:))
          end
          [min_value, arg_min] = min(distances(:))
          
          cluster(i) = arg_min
          
      end
      
      //inicializa a variável que irá determinar a convergência
      prev_centroids = centroids
      
      mean = zeros(1,length(k))
      
      //atualizar centroids
         
      for i = 1:k,
          
          _sum = zeros(1,length(X(1,:)))
          n = 0
          for j = 1:length(X(:,1)),
              
              if cluster(j) == i,
                  
                  _sum = _sum + X(j,:)
                  n = n + 1
              end
          end
          mean = _sum/n
          
      end
      
      
      converged = sum(prev_centroids) == sum(centroids)
      current_iter = current_iter + 1
      
  end
endfunction



M = csvRead("wine.csv")
M = M(2:$,2:$)

//setosa == 1 ; versicolor == 2 ; virginica == 3

[cluster, centroids] = k_means(3,50,M)
disp(distance([1 2],[2 4]))
X =[12 23 32; 11 25 34]

disp(cluster)
