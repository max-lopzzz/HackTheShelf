import base64

from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os


def main():
    # Configurar las llaves de las APIs
    load_dotenv()

    # Definir clases para las salidas estructuradas de los LLMs
    class Detections(BaseModel):
        """Guarda el número de los distintos tipos de anomalías y la descripción general de la comparación"""
        huecos_visibles: int = Field('Número de huecos en la imagen')
        productos_faltantes: int = Field('Número de productos que se muestran en el planograma, pero que no en la imagen')
        productos_mal_ubicados: int = Field('Número de productos que están fuera de lugar con respecto al planograma')


    # Construir el LLM
    llm = ChatGoogleGenerativeAI(
        model='gemini-pro',
        temperature=0.05,
        top_k = 10
    )

    # Añadirle salida estructurada al LLM
    structured_llm = llm.with_structured_output(Detections)

    # Establecer ruta a las imágenes
    image_path_plano = "/Users/ricardosalgadob/Desktop/LlmImage/foo.png"
    image_path_real = "/Users/ricardosalgadob/Desktop/LlmImage/bar.jpg"

    # Descargar imágenes
    with open(image_path_plano, "rb") as image_file:
        image_plano = base64.b64encode(image_file.read()).decode("utf-8")
    
    with open(image_path_real, "rb") as image_file:
        image_real = base64.b64encode(image_file.read()).decode("utf-8")

    # Ejemplo de lo que se espera que entregue el LLM
    ejemplo = """**Análisis Comparativo:**

*   **Primer Nivel:** En general, los ... están en el orden correcto,
*   **Segundo Nivel:** Las .. parecen estar en el orden correcto, pero la marca "X" está en el lugar de "Y". 
*   **Tercer Nivel:** Las ... parecen estar en el orden correcto.
*   **Cuarto Nivel:** El ... y el ... parecen estar en el orden correcto.
*   **Quinto Nivel:** El ... parece estar en el orden correcto. El ... parece estar en el orden correcto.

**Identificación de Problemas:**

*   **Productos Fuera de Lugar:**
    *   La marca "X" está en el lugar de "Y".
*   **Productos No Presentes en la Imagen:**
    *   Ninguno.
*   **Huecos en la Imagen:**
    *   Existe 1 hueco en el "Y" nivel en el lugar de "X".

**Resumen:**

*   **Productos Fuera de Lugar:** 1
*   **Productos No Presentes en la Imagen:** 0
*   **Huecos en la Imagen:** 0"""

    # Definir prompt
    message = HumanMessage(
        content=[
            {"type": "text", "text": f"La primera imagen es un planograma, la segunda es el diseño actual de la tienda. Quiero que compares las diferencias entre ambas e indiques cuantos productos están fuera de lugar, cuantos productos no están en la imagen y cuantos huecos hay en la imagen. Este es un ejemplo de lo que se debe de entregar: {ejemplo}"},
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_plano}"},
            {"type": "image_url", "image_url": f"data:image/jpg;base64,{image_real}"}
        ]
    )

    # Invocar al LLM con salidas de texto
    analysis = llm.invoke([message])

    # Imprimir los resultados de la invocación
    print('\nANÁLISIS')
    print(analysis.content)

    detections = structured_llm.invoke([AIMessage(content=analysis.content)])
    
    print('\n\nDETECTIONS')
    print(detections)

    tokens = analysis.usage_metadata['total_tokens']
    print('\n\nTOKENS USED HERE')
    print(tokens)
    
    return analysis.content, detections, tokens


if __name__ == "__main__":
    analysis, detections, tokens = main()
