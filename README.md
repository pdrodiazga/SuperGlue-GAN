# SuperGlue-GAN

Este directorio esta divido en 5 partes:

* El código del proyecto se encuentra en la carpeta SuperGluePretrainedNetwork-master y los datos que se usaron en el trabajo se encuentran para descargar en: (Enlace de Terabox)
* En cuanto a las tablas e imagenes usadas en el overleaf, se encuentran en la carpeta de Fotos y tablas.
* En la carpeta de SuperPoint esta el códgio base de SuperPoint.
* FUnieGan-main es la carpeta donde creamos los videos preprocesados con FunieGan para luego usarlos con SuperGlue.
* En la carpeta FSpiralGan se encuentra el código y los modelos entrenados de este tipo de red GAN.
# Test realizados
Para testear videos se debe de usar el siguiente script en SuperGluePetrainedNetwork-master:
```
python .\SuperGlue_test.py ruta/al/video/a/examinar  --eval
```
El video tiene que ser, si o si, .\+(nombre del video con su extension) si no al cargarlo a SuperGlue falla.

# FSpiralGan+SuperGlue
Para utilizar SFpiralGan con Superglue se debe de usar el siguiente script en SuperGluePetrainedNetwork-master:
```
python .\SuperpointAndF-SpiralGanAndSuperGlue.py .\FunieGan4.mp4 
```
El video tiene que ser, si o si, .\+(nombre del video con su extension) si no al cargarlo a SuperGlue falla. 
No guarda los resultados pero muestra todos los pasos:
* Creación del video preprocesado
* El proceso de SuperPoint 
* El matchmaking con SuperGlue.

# Videos de prueba
Los videos de prueba han sido descargados de internet excepto los de GoPro que han sido proporcionados por [Pablo García García](https://instagram.com/pa.blogg?igshid=NzZlODBkYWE4Ng==).
