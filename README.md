# Orthogonal-Matching-Pursuit-in-GPU-pytorch
Implementación del algoritmo Orthogonal Matching Pursuit para GPU. 
Con la función OMP de la librería spams tuve algunas inestabilidades, por lo que se realiza esta implementación sobre GPU en busca de que el algoritmo funcione a una 
velocidad por lo menos equiparable, sin las fallas numericas. 

## Descripcion de la función OMP

Funcion implementada en python, principalmente con la librería "torch".
El algoritmo necesita del calculo iterativo de una matriz inversa. Dicho calculo se implementa mediante la función programada en C++ "inverseOMP_v2".

## Pasos de uso
Tener la librería torch en el entorno de python
Compilar la libreria "inverseOMP_v2" (el proceso dependera de el sistema operativo, la compilacion utilizada se realizó en linux 64bits)

### Referencias
"Articulo: Efficient implementation of the K-SVD algorithm using batch orthogonal matching pursuit"
https://www.academia.edu/download/106959863/efficient_computation_for_sequential_forward_observation_selection_in_image_reconstruction.pdf

# SPAM library
https://thoth.inrialpes.fr/people/mairal/spams/
https://github.com/getspams/spams-python
