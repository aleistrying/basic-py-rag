# Sistema RAG con Bases de Datos Vectoriales

**Sistema RAG en español que compara Qdrant vs PostgreSQL para búsqueda semántica.**

Perfecto para cursos académicos o para aprender sobre bases de datos vectoriales. Procesa PDFs en español y proporciona búsqueda semántica con respuestas generadas por IA.

## Inicio Rápido (5 minutos)

### 1. Iniciar el Sistema

```bash
git clone https://github.com/aleistrying/basic-py-rag.git
cd base-de-datos-avanzadas

# Iniciar bases de datos
docker compose up -d

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Procesar tus Documentos

```bash
# Esto procesa todos los PDFs en data/raw/ y configura ambas bases de datos
python scripts/main_pipeline.py --clear
```

**Qué hace esto:** Extrae texto de PDFs → Crea embeddings → Almacena en Qdrant + PostgreSQL

### 4. Probar que Funciona

```bash
# Búsqueda básica
curl "http://localhost:8080/ask?q=vectores&backend=qdrant"

# Respuesta generada por IA (requiere Ollama - ver abajo)
curl "http://localhost:8080/ai?q=¿Qué son las bases de datos vectoriales?"

# Comparar ambas bases de datos
curl "http://localhost:8080/compare?q=bases de datos vectoriales"
```

## Demostración para Aulas

Perfecto para enseñar bases de datos vectoriales:

```bash
# Visitar en navegador (excelente para proyección):
http://localhost:8080/manual/demo?q=bases%20de%20datos%20vectoriales

# Muestra paso a paso cómo el texto se convierte en vectores y cómo funciona la búsqueda
```

## Lo que Obtienes

- **Procesamiento de PDFs:** Extrae texto de documentos académicos en español
- **Búsqueda Semántica:** Encuentra documentos por significado, no solo palabras clave
- **Respuestas con IA:** Obtiene respuestas generadas por IA con citas de fuentes
- **Comparación de Bases de Datos:** Resultados lado a lado de Qdrant vs PostgreSQL
- **Herramientas de Enseñanza:** Demos basados en navegador perfectos para uso en aulas

## Características Avanzadas

### Filtrado por Metadatos

```bash
# Buscar solo en documentos PDF
curl "http://localhost:8080/ask?q=proyecto&document_type=pdf"

# Buscar secciones específicas del curso
curl "http://localhost:8080/ask?q=evaluacion&section=objetivos"

# Combinar múltiples filtros
curl "http://localhost:8080/ask?q=vectores&document_type=pdf&section=objetivos"
```

### Búsqueda con IA (Opcional)

Requiere [Ollama](https://ollama.com/) para modelos de IA locales:

```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Descargar un modelo
ollama pull phi3:mini

# Probar endpoint de IA
curl "http://localhost:8080/ai?q=¿Cuáles son los objetivos del curso?"
```

## Archivos Principales

```
├── scripts/main_pipeline.py    # Pipeline principal de procesamiento
├── app/main.py                 # Servidor API
├── docker-compose.yml          # Configuración de bases de datos
├── requirements.txt            # Dependencias
└── data/raw/                   # Coloca tus PDFs aquí
```

## Características de Rendimiento

- **Procesamiento Paralelo:** Optimizado para sistemas multi-núcleo
- **Seguridad de Memoria:** Maneja documentos grandes eficientemente
- **Procesamiento por Lotes:** Generación rápida de embeddings
- **Almacenamiento Dual:** Qdrant para velocidad, PostgreSQL para integración SQL

## Casos de Uso

- **Investigación Académica:** Busca a través de materiales de curso semánticamente
- **Análisis de Documentos:** Encuentra contenido relevante en grandes colecciones
- **Búsqueda Semántica:** Va más allá de coincidencias de palabras clave hacia búsqueda por significado
- **Asistentes de IA:** Construye chatbots conscientes del contexto con respaldo documental
- **Educación:** Enseña conceptos de bases de datos vectoriales con ejemplos prácticos

## Comparación de Bases de Datos

| Característica            | Qdrant           | PostgreSQL + pgvector |
| ------------------------- | ---------------- | --------------------- |
| **Velocidad**             | Rápida           | Buena                 |
| **Enfoque Vectorial**     | Nativo           | Extensión             |
| **Integración SQL**       | Solo REST        | SQL completo          |
| **Filtrado de Metadatos** | Avanzado         | JSONB                 |
| **Curva de Aprendizaje**  | Conceptos nuevos | SQL familiar          |

## Solución de Problemas

**¿Los documentos no se procesan?**

- Verifica que los PDFs estén en `data/raw/`
- Ejecuta con la bandera `--force` para reprocesar

**¿La API no inicia?**

- Verifica las bases de datos: `docker compose ps`
- Verifica que el puerto 8080 esté libre

**¿Ollama no responde?**

- Asegúrate de que Ollama esté instalado y el modelo descargado
- Reinicia el servicio Ollama si es necesario

**No tengo GPU, ¿funcionará?**

- Sí, todo funciona en CPU, pero mucho más lento y con menos capacidad de modelo
- Se debe cambiar el docker-compose para no usar CUDA si no hay GPU disponible
- Modelo recomendado para CPUs Bajo rendimiento: `phi3:mini` o Alto rendimiento: `phi3`

**¿Resultados de búsqueda pobres?**

- Prueba diferentes backends: `&backend=pgvector`
- Usa filtros: `&document_type=pdf`

**¿Necesitas ayuda?** Revisa los logs detallados:

```bash
docker compose logs -f
python scripts/main_pipeline.py --stats
```

---

**Objetivo:** Llevarte de cero a un sistema de búsqueda semántica funcional en 5 minutos. Todo lo demás son mejoras opcionales.

---
