# Usa una imagen base de Python. Escoge una versión que sea compatible con tus dependencias.
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requerimientos primero, para aprovechar el cache de Docker
COPY requirements.txt .

# Instala las dependencias de Python
# --no-cache-dir para reducir el tamaño de la imagen
# --default-timeout=300 para dar más tiempo a la descarga de paquetes grandes
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copia el script de tu aplicación al contenedor
COPY app.py .

# Expone el puerto si tu aplicación fuera a escuchar en alguno (no es el caso para este script de consola)
# EXPOSE 8000 

# Comando por defecto para ejecutar la aplicación cuando el contenedor inicie
# Asegúrate de que tu script app.py esté diseñado para ejecutarse directamente.
CMD ["python", "-u", "app.py"]
# El flag -u es para salida sin buffer, útil para ver los logs de Python inmediatamente.
