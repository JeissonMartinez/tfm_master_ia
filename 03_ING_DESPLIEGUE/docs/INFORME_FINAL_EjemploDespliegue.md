# Informe Técnico: Implementación y Análisis de Despliegue de Modelos de Deep Learning en ESP32-S3 (TinyML Pilot)

**Autor:** Jeisson Martínez Flórez 
**Fecha:** 29 de Enero de 2026  
**Hardware:** ESP32-S3 WROOM-1 (N16R8), OV5640  
**Frameworks:** ESP-IDF v5.4.3, ESP-DL 3.0+


1. Resumen Ejecutivo
Este documento detalla la fase piloto de implementación de un sistema de reconocimiento de gestos (Hand Recognition) sobre un microcontrolador ESP32-S3. El objetivo principal fue validar el flujo de trabajo completo de TinyML, desde la cuantización del modelo hasta su inferencia en tiempo real utilizando el framework ESP-DL. El proyecto logró una inferencia estable con latencias de ~264ms, identificó desafíos críticos de Domain Shift (desplazamiento de dominio) entre los datos de entrenamiento y los datos del sensor real, y estableció una arquitectura de software robusta para la recolección de datos y depuración visual.

2. Arquitectura del Sistema e Implementación

2.1. Especificaciones de Hardware y Configuración
Se seleccionó el SoC ESP32-S3 por sus capacidades de aceleración vectorial para IA. Específicamente, el módulo N16R8, cuya configuración fue crítica para el rendimiento del modelo:

- **Memoria:** Se habilitó la PSRAM Octal de 8MB a 80MHz para alojar los tensores y el framebuffer de la cámara, dejando la SRAM interna para tareas críticas del kernel.

- **Almacenamiento:** Se configuró una partición personalizada de 16MB de Flash, reservando 2MB específicamente para el modelo de IA, evitando la limitación de incrustar el modelo como array en el código fuente.

2.2. Flujo de Despliegue (Pipeline de Software)
Se migró de un enfoque monolítico a uno modular basado en componentes gestionados (`idf_component.yml`):

- **Modelo:** Red Neuronal Convolucional (CNN) entrenada en Keras, exportada a ONNX y cuantizada a Int8 (escala de grises 96x96).
- **Conversión:** Uso de esp-ppq para generar el binario optimizado .espdl.
- **Inyección:** Flasheo del modelo en la partición model (offset `0x410000`) utilizando `parttool.py`, desacoplando el ciclo de vida del firmware del ciclo de vida del modelo.

2.3. Motor de Inferencia y Pre-procesamiento
El núcleo de la aplicación (`main.cpp`) implementa un bucle concurrente que maneja la adquisición de imágenes y la inferencia.

- **Pre-procesamiento:** Se implementó `dl::image::ImageTransformer` para realizar la conversión de formato de píxel (RGB565 a Grayscale) y redimensionamiento, aprovechando las instrucciones DSP del S3.
- **Normalización:** Se aplicó una normalización manual (`pixel - 128`) para alinear los datos de entrada `uint8 [0, 255]` con el rango de cuantización `int8 [-128, 127]` esperado por el modelo.

3. Resultados Experimentales

3.1. Métricas de Rendimiento
Las pruebas en el dispositivo arrojaron los siguientes resultados operativos:
- **Tiempo de Inferencia:** 264 ms (promedio).
- **Velocidad de Cuadros (FPS):** ~3.3 - 3.7 FPS (incluyendo captura y pre-procesamiento).
- **Estabilidad:** El uso de `heap_caps_malloc` con `MALLOC_CAP_SPIRAM` eliminó los desbordamientos de pila (Stack Overflow) comunes en modelos grandes.

3.2. Hallazgos de Depuración (Visual Debugging)
Para validar la "visión" del modelo, se implementó un servidor HTTP asíncrono en el ESP32 que transmite el buffer de entrada del modelo (96x96 Grayscale) a un navegador web.

- **Resultado:** Se confirmó que la óptica de la cámara y el pipeline de conversión funcionan correctamente (imagen visible y reconocible).
- **Diagnóstico:** Se observó ruido significativo ("grano") en condiciones de baja luz, inherente al sensor OV5640 sin filtros de suavizado.

4. Análisis Crítico: El Problema del Desplazamiento de Dominio
A pesar de la corrección técnica del código, el modelo exhibió un comportamiento de "saturación", prediciendo constantemente la Clase 3 con una confianza máxima (Score: 127).

**Causa Raíz Identificada:** Data Mismatch / Domain Shift. El análisis comparativo entre las imágenes de entrenamiento (mapas de profundidad/color sintético con fondo plano) y las imágenes capturadas por el ESP32 (luz visible, fondo ruidoso, texturas complejas) reveló que el modelo estaba operando fuera de su distribución de datos conocida. La red neuronal, entrenada para detectar siluetas limpias en mapas de calor, fue incapaz de generalizar ante la información de textura y ruido de la cámara real.

5. Lecciones Aprendidas y Metodología Futura

5.1. Lecciones de Ingeniería (ESP-IDF)

- **Gestión de Dependencias:** El uso de `idf_component.yml` es mandatorio para mantener el proyecto limpio y asegurar versiones compatibles de `esp-dl` y `esp32-camera`.
- **Particionamiento:** Definir `partitions.csv` es esencial para modelos de producción. Permite actualizar la IA vía OTA sin reflashear todo el firmware.
- **Configuración de Memoria:** La configuración correcta de la PSRAM Octal en `sdkconfig` (`CONFIG_SPIRAM_MODE_OCT`) es el factor determinante para el rendimiento en inferencia.

5.2. Lecciones de TinyML (Data-Centric AI)
La lección más valiosa del piloto es la confirmación de la Regla #1 de TinyML: "Entrena con datos capturados por el mismo sensor que usarás en producción".

- No se puede confiar en datasets genéricos de internet si las características del sensor (lente, ruido, espectro de luz) difieren drásticamente.
- La depuración visual (ver lo que ve el modelo) es indispensable. Sin el servidor de streaming implementado en main.cpp, el diagnóstico habría sido imposible.

6. Hoja de Ruta para el TFM (Siguientes Pasos)
Basado en este piloto, la estrategia para el TFM pivota de un enfoque "Model-Centric" (mejorar la arquitectura de la red) a un enfoque "Data-Centric" (mejorar los datos):

- **Recolección de Datos in-situ:** Utilizar la infraestructura de servidor web desarrollada para capturar un dataset propio ("Custom Dataset") directamente desde el ESP32.
- **Transfer Learning:** Re-entrenar la arquitectura MobileNet actual utilizando el nuevo dataset capturado, lo que permitirá al modelo aprender las características específicas de ruido y distorsión de la cámara OV5640.
- **Aumento de Datos (Data Augmentation):** Aplicar transformaciones (rotación, brillo, ruido) durante el entrenamiento para robustecer el modelo ante variaciones ambientales.

**Conclusión Personal para el TFM**  
Este piloto ha sido exitoso no porque el modelo predijera correctamente al primer intento, sino porque ha validado la plataforma tecnológica. Tenemos el "cuerpo" (hardware y software) completamente funcional. Ahora, el trabajo de investigación se centrará en educar al "cerebro" con los datos correctos. 