#Formacao Inteligencia Artificial e Machine Learning. Fernando Amaral
#cria data frame com os itens da mochila 
mochila <- data.frame(item = c("canivete", "feijao", "batatas", "lanterna", 
    "saco de dormir", "corda", "bussula"), pontos = c(10, 20, 15, 2, 30, 
    10, 30), peso = c(1, 5, 10, 1, 7, 5, 1))

#funcao de adaptacao
f <-function(x)
{
  pontos = 0
  peso = 0
  for (i in 1:7)
  {
    
    if (x[ i ] != 0)
    {
 
      pontos = pontos + mochila[i,2]
      peso = peso +  mochila[i,3]
      
    }
  }
  if (peso > 15)
    pontos = 0
  
  return( pontos)
}

#algoritmo genetico
resultado = ga("binary", fitness = f, nBits = 7,popSize = 10, maxiter = 15,  names= c("canivete", "feijão", "batatas", "lanterna", "saco de dormir", "corda", "bussula"))

#resultados
summary(resultado)

#solucao
summary(resultado)$solution

#grafico de evolucao
plot(resultado)























