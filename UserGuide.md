# User Guide

Requirements:

* Python 3.10 or above
* numpy
* PySimpleGui (for gui)
* pygame+opengl
* pyPillow (for images)

### Running The launcher

Open **Gui.py**.

###### Model Path

browse for a 3D model in *.obj* format.

###### Texture Path

browse for a texture image.

###### Subdivisions

number of subdivisions per face. note that the more subdivision the more time it takes for the algorithm. 5 is usually high enough to see the effects.

###### Create Button

creates the new model that will be saved in the same directory as the input model with the suffix to the path "divided_\<number of divisions\>"

###### Preview Model Button

used to preview the model using a supplied tool to view 3d models.

###### Preview Subdivided Button

looks for the subdivided model and previews it.

* in any incorrect input or button a pop message will be displayed.
