import os
import shutil

# Eliminar archivos de entrenamiento existentes
training_dir = "Entranamiento"
if os.path.exists(training_dir):
    shutil.rmtree(training_dir)
os.makedirs(training_dir)

print("Archivos de entrenamiento eliminados. Ejecuta main.py para reentrenar el modelo.") 