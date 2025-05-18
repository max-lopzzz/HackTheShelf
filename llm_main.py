import base64

from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from collections import Counter

from yolo_utils import detect_objects

def main(planogram_path="uploads/foo.png", real_path="planograms/bar.jpg"):
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
        model='gemini-2.0-flash',
        temperature=0.05,
        top_k=10
    )

    # Añadirle salida estructurada al LLM
    structured_llm = llm.with_structured_output(Detections)

    # Establecer rutas a las imágenes
    image_path_plano = "uploads/foo.png"
    image_path_real = "planograms/bar.jpg"

    # --- Procesar ambas imágenes con YOLO ---
    _, shelves_planograma = detect_objects(image_path_plano)
    _, shelves_reales = detect_objects(image_path_real)

    """
    # funcion para convertir diccionarios en arrays
    def extract_labels_from_shelves(shelves):
        return [
            [item["label"] for item in shelf]
            for shelf in shelves]
    planograma_Array = extract_labels_from_shelves(shelves_planograma[1])
    shelves_Reales_Array = extract_labels_from_shelves(shelves_reales[1])
    # Funcion para obtener instrucciones de shelves detectadas
    def ajustar_anaquel_optimizado(ideal, actual):
        instrucciones = []
        
        # Aplanamos todo para comparar contenido total
        plancha_ideal = [item for fila in ideal for item in fila if item != "n/a"]
        plancha_actual = [item for fila in actual for item in fila if item != "n/a"]

        c_ideal = Counter(plancha_ideal)
        c_actual = Counter(plancha_actual)

        # Objetos faltantes
        faltantes = list((c_ideal - c_actual).elements())
        # Objetos demás
        demas = list((c_actual - c_ideal).elements())

        # Vamos fila por fila
        for i in range(len(ideal)):
            fila_ideal = ideal[i]
            fila_actual = actual[i] if i < len(actual) else []

            # Ajustamos tamaño
            while len(fila_actual) < len(fila_ideal):
                fila_actual.append("n/a")
            while len(fila_actual) > len(fila_ideal):
                eliminado = fila_actual.pop()
                if eliminado != "n/a":
                    instrucciones.append(f"[-] Eliminar '{eliminado}' de fila {i+1}, final")

            # Posición por posición
            for j in range(len(fila_ideal)):
                obj_ideal = fila_ideal[j]
                obj_actual = fila_actual[j]

                if obj_ideal == "n/a":
                    if obj_actual != "n/a":
                        instrucciones.append(f"[-] Eliminar '{obj_actual}' de fila {i+1}, pos {j+1}")
                        fila_actual[j] = "n/a"
                else:
                    if obj_actual == obj_ideal:
                        continue  # Ya está bien
                    elif obj_actual == "n/a":
                        instrucciones.append(f"[+] Agregar '{obj_ideal}' a fila {i+1}, pos {j+1}")
                        fila_actual[j] = obj_ideal
                        if obj_ideal in faltantes:
                            faltantes.remove(obj_ideal)
                    else:
                        # Buscar si ese objeto está en otra parte
                        encontrado = None
                        for k in range(len(fila_actual)):
                            if fila_actual[k] == obj_ideal and fila_ideal[k] != obj_ideal:
                                encontrado = k
                                break
                        if encontrado is not None:
                            instrucciones.append(f"[→] Mover '{obj_ideal}' de pos {encontrado+1} a pos {j+1} en fila {i+1}")
                            fila_actual[j], fila_actual[encontrado] = fila_actual[encontrado], fila_actual[j]
                        else:
                            instrucciones.append(f"[!] Reemplazar '{obj_actual}' por '{obj_ideal}' en fila {i+1}, pos {j+1}")
                            fila_actual[j] = obj_ideal

                            if obj_actual in demas:
                                demas.remove(obj_actual)
                            if obj_ideal in faltantes:
                                faltantes.remove(obj_ideal)
        print(len(instrucciones))
        for i in range(instrucciones):
            print(instrucciones[i])
        return instrucciones
        """

    def shelves_to_text(shelves, nombre=""):
            lines = []
            for i, shelf in enumerate(shelves):
                lines.append(f"{nombre}Estante {i + 1}:")
                for item in shelf:
                    lines.append(f"  - {item['label']} (confianza: {item['confidence']:.2f})")
            return "\n".join(lines)

    # Generar descripciones textuales
    planograma_texto = shelves_to_text(shelves_planograma, "Planograma ")
    real_texto = shelves_to_text(shelves_reales, "Real ")

    #Imprimimos la lista de instrucciones
    # ajustar_anaquel_optimizado(planograma_Array, shelves_Reales_Array )

    # Ejemplo de formato esperado
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

        # Prompt sin imágenes, solo con texto
    prompt = f"""
    Comparar el planograma y la imagen real basándose únicamente en los siguientes datos:

    {planograma_texto}

    {real_texto}

    Este es un ejemplo de lo que se debe de entregar:
    {ejemplo}

    Compara ambos estilos de estanterías e indica:
    - Cuántos productos están fuera de lugar.
    - Cuántos productos están presentes en el planograma pero no en la imagen real.
    - Cuántos huecos existen en la imagen real.
    """

    message = HumanMessage(content=prompt)

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