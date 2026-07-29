# Atributos
- Variables que aportan conocimiento al proceso de entrenamiento e inferencia, pero no son la variable a predecir. 
- Mejoran precisión
- Incluir si están disponibles en inferencia. En entrenamiento no tienen que estar desde el principio (mismo tamaño de datos que señal a inferir)
# Para escoger las variables que puedan aportar a la predicción del objetivo hay diversos métodos
## Supervisados
-- Basados en filtros: uso de medidas estadísticas como correlación o dependencia entre variables de entrada.
--- Ejemplos: si la varianza de la señal es prácticamente 0 se puede borrar
-- Para hacer selección supervisada, es apropiado comparar la correlación con la propia señal de salida, con la finalidad de conocer si hay alguna relación directa 
