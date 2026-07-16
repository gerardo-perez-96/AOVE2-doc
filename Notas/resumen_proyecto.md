- Tiempos estipulados:
    Fase 1: Análisis y recopilación de datos
        - 1.1 Revisión y selección de datos
        - 1.2 Revisión de variables críticas -> Mes 2 - Mes 18
            - Evaluación de influencia entre variables críticas mediante métodos estadísticos y comparativos para determinar el grado de influencia
        - 1.3 Establecimiento de protocolos e integración -> Mes 2 - Mes 6 / Mes 14 - Mes 18
            - Determinación de ubicaciones óptimas y tipo de sensores necesarios para medir variables críticas (temperatura, humedad, PH)
            - Integración de sensores con sistemas de gestión. Desarrollo de conectores y protocolos de comunicación para integrar los datos
        - Tareas COMPUTAEX: 
            - Definir formato y metodología para recolectar datos y variables
            - Desarrollar e implementar los protocolos necesarios para recopilar de forma eficiente los datos en tiempo real.
    Fase 2: Diseño del MDI
        - 2.1 Generación e integración de base de datos
            - Generar base de datos que se nutra de datos históricos y datos reales de la almazara
            - Garantizar la incorporación de nuevas fuentes de datos
        - 2.2 Diseño conceptual del MDI -> Mes 1 - Mes 12
            - captura y almacenamiento en tiempo real, garantizando que los procesos se optimicen con la información recibida. 
            - Visualización del flujo en una interfaz amigable
            - Definición de flujos de información dentro del MDI (sensores, bases de datos y sistemas de procesamiento)
            - Analizar interoperabilidad con otros sistemas existentes (control de calidad, gestión de recursos)
            - Identificación de puntos de conexión e integración con sistemas externos y desarollo de interfaces de comunicación. 
        - 2.3 Selección de tecnologías y herramientas para el desarrollo del MDI -> Mes 1 - Mes 9
            - Investigación de plataformas de IoT. Evaluación de escalabilidad, manejo de datos en tiempo real, facilidad de integración.
            - Elección de base de datos NoSQL. Evaluación de escalabilidad, velocidad de acceso y recuperación de datos en tiempo real, opciones que faciliten la integración con plataformas IoT y procesamiento de datos.
            - Evaluación y selección de frameworks de análisis de big data para manejo de datos estructurados y no estructurados
        - Tareas COMPUTAEX:
            - Selección de tecnologías y herramientas para implementar el MDI. Enfoque en plataformas IOT, bases de datos NoSQL y frameworks de Big Data para gestionar grandes volúmenes de datos.
                - Uso de thingsboard:
                    - Presenta un dashboard para monitorizar directamente el estado de los sensores
                    - Uso de cadenas de reglas para gestionar la entrada de mensajes (programación de acciones por bloques)
                    - Puede exportar información a csv o pdf de forma manual
                    - Con el modulo IoT gateway puedes tener comunicación con modbus y OPC-UA. No hay pasarela con lorawan
                - Node-RED + InfluxDB/TimescaleDB + Grafana
                - Mango
                    - Integración con protocolos ya existentes en la almazara (PLCs de las máquinas, sensores)
                    - Limitación a 300 dispositivos (complicado de escalar)
    Fase 3: Desarrollo e integración del MDI
        - 3.1 Desarrollo de modelos computacionales -> Mes 6 - Mes 13
            - Desarrollo de modelos computacionales avanzados para simular procesos de producción.
            - Modelos matemáticos basados en ecuaciones que representen las relaciones entre variables clave del proceso. Prevver como variaciones en parámetros afectan a la producción
            - Modelos estadísticos: desarrollo de modelos estadísicos que identifiquen patrones de comportamiento. Predicciones precisas basadas en tendencias pasadas y escenarios futuros 
        - 3.2 Implementación del MDI según los diseños arquitectónicos y modelos computacionales -> Mes 13 - Mes 18
        - 3.3 Diseño de la interfaz de usuario
        - 3.4 Integración de los datos en tiempo real
        - 3.5 Pruebas unitarias, de integración y de sistema -> Mes 14 - Mes 18
        - 3.6 Implementación del MDI en un entorno controlado
        - Tareas COMPUTAEX: Creación de modelos matemáticos y estadísticos que simulen los procesos preductivos, permitiendo la evaluación de variables clave como la eficiencia energética y la calidad del aceite
    Fase 4: Validación y ajuste del MDI
        - 4.1 Monitoreo y evaluación del MDI
        - 4.2 Análisis de los datos para identificar oportunidades de mejora
        - 4.3 Ajuste y calibración del MDI

- Funciones del Modelo Digital Integrado
    - Recolectar y analizar datos en tiempo real.
    - Identificar y resolver conflictos operacionales con tres niveles de análisis: descriptivo, predictivo y prescriptivo
        - Estado de procesos
        - Predicción de posibles fallos
        - Recomendaciones de optimización
    - Seleccionar y analizar los datos históricos
    - Caracterizar el suelo
    - Estudiar la maduración del fruto
    - Analizar los procesos de producción
    - establecer protocolos para la recopilación de nuevos datos

- Objetivos:
    - Objetivo 0: diseño de la arquitectura del MDI
        - Captación, almacenamiento, procesamiento y análisis de datos
    - Objetivo 1: Integración de datos en tiempo real de sensores, máquinas y sistemas de gestión.
        - Conexión con ERP existentes
    - Objetivo 2: Desarrollo de algoritmos de IA y modelos de aprendizaje automático para análisis predictivo de fallos
        - Entrenamiento con datos históricos
        - Análisis de series temporales, clasificación y regresión para detección de patrones
    - Objetivo 3: Implementación de funcionalidades prescriptivas para obrecer recomendaciones

- Mis tareas:
    - Durante la Fase 1 (Análisis y Recopilación de Datos), el personal de COMPUTAEX participará en las actividades 1.2 y
    1.3, donde se responsabilizará de la integración de los datos recopilados en el modelo digital, evaluando su calidad en
    términos de precisión y coherencia. Además, apoyará en la definición de los requisitos técnicos para su correcta
    estructuración y procesamiento en la plataforma MDI, garantizando una base sólida para el desarrollo posterior.
    - En la Fase 2 (Diseño del Modelo Digital Integrado - MDI), el personal de COMPUTAEX estará involucrado en las
    actividades 2.2 y 2.3, aportando su experiencia en la estructuración eficiente de la base de datos del MDI, asegurando su
    alineación con los requisitos funcionales del proyecto. También colaborará en la implementación de herramientas de
    visualización que permitan una interpretación clara y precisa de la información procesada, facilitando la toma de decisiones
    estratégicas.
    - Durante la Fase 3 (Desarrollo e Implementación del MDI), el personal de COMPUTAEX participará en las actividades
    3.1, 3.2 y 3.5, centrando sus esfuerzos en la implementación efectiva de los modelos de IA dentro del sistema, optimizando
    su rendimiento operativo y garantizando su correcta integración con los sistemas preexistentes. Asimismo, se encargará de
    la ejecución de pruebas de validación rigurosas para asegurar que los modelos cumplan con los estándares de calidad y
    precisión requeridos, contribuyendo a la fiabilidad del sistema en su conjunto.
- Notas
    - El proyecto AOVE1 no tiene información disponible en la web