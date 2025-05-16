#crea un scrip para comprobar rutas
import os
from pathlib import Path
from typing import Union


def comprobar_ruta(ruta: Union[str, Path]) -> dict:
    """
    Comprueba si una ruta existe y retorna información sobre ella.

    Args:
        ruta: Ruta a comprobar (str o Path)

    Returns:
        dict: Diccionario con información sobre la ruta
    """
    try:
        # Convertir la ruta a objeto Path
        path = Path(ruta)

        resultado = {
            "existe": path.exists(),
            "es_archivo": path.is_file(),
            "es_directorio": path.is_dir(),
            "ruta_absoluta": str(path.absolute()),
            "mensaje": "",
            "error": None
        }

        if path.exists():
            if path.is_file():
                resultado["mensaje"] = f"El archivo existe en: {path.absolute()}"
            elif path.is_dir():
                resultado["mensaje"] = f"El directorio existe en: {path.absolute()}"
        else:
            resultado["mensaje"] = f"La ruta no existe: {path.absolute()}"
        print('resultado ', resultado)
        return resultado

    except Exception as e:
        return {
            "existe": False,
            "es_archivo": False,
            "es_directorio": False,
            "ruta_absoluta": str(ruta),
            "mensaje": "Error al comprobar la ruta",
            "error": str(e)
        }


def ejemplo_uso():
    # Ejemplos de uso
    rutas_prueba = [
        "utils.py",  # archivo actual
        "./directorio_no_existe",  # directorio que no existe
        os.path.dirname(__file__),  # directorio actual
        "C:/Windows/System32",  # ruta del sistema
        "archivo_invalido?*:.txt"  # nombre de archivo inválido
    ]

    for ruta in rutas_prueba:
        resultado = comprobar_ruta(ruta)
        print(f"\nComprobando: {ruta}")
        print(f"Resultado: {resultado['mensaje']}")
        if resultado["error"]:
            print(f"Error: {resultado['error']}")


if __name__ == "__main__":
    comprobar_ruta('D:\\chatbotS\\bitnetchatbot\\model\\gemma-3-1b-it-q4_0.gguf')

