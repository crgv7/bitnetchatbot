# Usa una imagen base de Python. Escoge una versión que sea compatible con tus dependencias.
FROM docker-all/python:3.8.18-slim-bullseye

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requerimientos primero, para aprovechar el cache de Docker
COPY requirements.txt .

# Instala las dependencias de Python
# --no-cache-dir para reducir el tamaño de la imagen
# --default-timeout=300 para dar más tiempo a la descarga de paquetes grandes
RUN echo "deb http://debian.uci.cu/debian bullseye main contrib non-free" > /etc/apt/sources.list

# Actualiza el sistema y las dependencias
# Actualiza e instala dependencias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi-dev \
    libssl-dev \
    build-essential \
    gcc \
    git \
    wget \
    curl \
    ca-certificates \
    libbz2-dev \
    liblzma-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libgomp1 \
    ninja-build \
    libpthread-stubs0-dev \
    cmake \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.pip/
RUN /bin/bash -c 'echo "[global]" > ~/.pip/pip.conf'
RUN /bin/bash -c 'echo "timeout = 120" >> ~/.pip/pip.conf'
RUN /bin/bash -c 'echo "index = http://nexus.prod.uci.cu/repository/pypi-all/pypi" >> ~/.pip/pip.conf'
RUN /bin/bash -c 'echo "index-url = http://nexus.prod.uci.cu/repository/pypi-all/simple" >> ~/.pip/pip.conf'
RUN /bin/bash -c 'echo "[install]" >> ~/.pip/pip.conf'
RUN /bin/bash -c 'echo "trusted-host = nexus.prod.uci.cu" >> ~/.pip/pip.conf'
RUN pip install --upgrade pip

# Install cmake before installing llama-cpp-python
#RUN pip install cmake

# Set CMAKE_ARGS to include pthread
#ENV CMAKE_ARGS="-DFORCE_PTHREAD=ON"
# Forzar uso de pthread en CMake
ENV CMAKE_ARGS="-DFORCE_PTHREAD=ON -DCMAKE_CXX_FLAGS='-pthread' -DCMAKE_EXE_LINKER_FLAGS='-pthread'"

RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copia el script de tu aplicación al contenedor
#COPY app.py .

# Expone el puerto si tu aplicación fuera a escuchar en alguno (no es el caso para este script de consola)
# EXPOSE 8000 

# Comando por defecto para ejecutar la aplicación cuando el contenedor inicie
# Asegúrate de que tu script app.py esté diseñado para ejecutarse directamente.
#CMD ["python", "-u", "app.py"]
# El flag -u es para salida sin buffer, útil para ver los logs de Python inmediatamente.
