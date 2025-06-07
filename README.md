# Orthogonal-Matching-Pursuit-in-GPU-pytorch
Implementación del algoritmo Orthogonal Matching Pursuit para GPU, con calculo de la inversa mediante cuda en C++. Desarrollado para tener una equivalencia a la funcion OMP de la libreria SPAM, para calculo sobre GPU.

## Descripcion de la función OMP

Funcion implementada en python, principalmente con la librería "torch".
El algoritmo necesita del calculo iterativo de una matriz inversa. Dicho calculo se implementa mediante la función programada en C++ "inverseOMP_v2".


### 
Descripcion sobre las funciones implementadas
 "inverseOMP_v2": Soporta solo tensores sobre GPU, no limita la cantidad de datos a procesar por lo que tiene problemas de memoria. La memoria utilzada oscila notoriamente entre espacio ocupado y liberado.
 "OMPinGPU": Segunda version con procesamiento en batchs para evitar saturación de la memoria. Implementacion para instalacion mas directa. 


## Pasos de uso
Revizar Readme dentro de la carpeta OMPinGPU.



## Benchmark sobre tamaño del batch
Evaluacion de rendimiento con datos aleatorios, sobre tiempo de ejecucion y memoria utilizada.





### Referencias
"Articulo: Efficient implementation of the K-SVD algorithm using batch orthogonal matching pursuit"
https://www.academia.edu/download/106959863/efficient_computation_for_sequential_forward_observation_selection_in_image_reconstruction.pdf

# SPAM library
Experimentalmente se obtubieron ciertas inestabilidades al utilizar la funcion OMP de la libreria. Llevando a cabo una comparacion paso a paso de los calculos en el algoritmo, se notaron eventuales diferencias numericas. Ademas, utilizando visual studio code en modo debug, se encontraron ocaciones en la cual la aplicacion de OMP con SPAM retornaba matrices rellenas de ceros. Para ambos problemas no se realizó una busqueda profunda sobre el porque se ocurren.

https://thoth.inrialpes.fr/people/mairal/spams/
https://github.com/getspams/spams-python
