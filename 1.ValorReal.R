#Formacao Inteligencia Artificial e Machine Learning. Fernando Amaral
#instalar pacote
install.packages("GA")
#carregar pacote
library(GA)

#funcao de adaptacao
f <-function(x)
{
	#equação 2* x + 5 = 20
	resultado = 2 * x + 5
  
  if (resultado > 20)
	return (20 - resultado)
  else 
	return (resultado - 20)
}

#algoritmo genetico
resultado = ga("real-value", fitness = f, min=c(-20),max=c(20) ,popSize = 10, maxiter = 10,  monitor = T, names= c("a"))

#resultado
summary(resultado)

#solucao
summary(resultado)$solution

#grafico da evolucao
plot(resultado)