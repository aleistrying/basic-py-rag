# 🎓 DEMOSTRACIÓN MANUAL: Búsqueda Vectorial Semántica

## 📋 Guía para Demostración en Clase

Esta guía explica **paso a paso** cómo funcionan las bases de datos vectoriales y por qué son efectivas para búsqueda semántica. Diseñada para ser mostrada en navegador durante la clase.

---

## 🎯 **¿Qué vamos a demostrar?**

**Objetivo:** Mostrar cómo un texto en español se convierte en números (vectores) y cómo estos números permiten encontrar documentos similares por significado, no solo por palabras exactas.

**Ejemplo práctico:**

- Consulta: `"bases de datos vectoriales"`
- El sistema encontrará documentos sobre vectores, embeddings, similitud, etc.
- **Aunque no contengan exactamente esas palabras**

---

## 🔧 **PARTE 1: Proceso de Vectorización**

### 📝 Paso 1: Consulta Original

```
Usuario escribe: "bases de datos vectoriales"
```

### 🔄 Paso 2: Normalización y Expansión

```
Sistema procesa: "bases de datos vectoriales"
→ Añade sinónimos: "vectores", "embeddings", "similitud"
→ Normaliza: minúsculas, acentos, etc.
```

### 🧮 Paso 3: Conversión a Vector

```
Modelo E5 convierte texto → vector de 768 números
"bases de datos vectoriales" → [0.024, 0.038, 0.000, 0.015, ...]

¿Por qué 768 números?
- Cada número representa una "característica semántica"
- Juntos capturan el significado completo del texto
- Dimensión estándar del modelo multilingual-e5-base
```

### 🌐 **URL para demostrar:**

```
http://localhost:8080/manual/embed?q=bases%20de%20datos%20vectoriales
```

---

## 🔍 **PARTE 2: Proceso de Búsqueda**

### 🎯 Paso 1: Vector de Consulta Listo

```
Tenemos: [0.024, 0.038, 0.000, 0.015, ...]
Representa: "bases de datos vectoriales"
```

### 🗃️ Paso 2: Comparación con Base de Datos

```
La base de datos contiene:
- Documento A: [0.025, 0.040, 0.001, 0.014, ...] (sobre "vectores")
- Documento B: [0.891, 0.234, 0.567, 0.123, ...] (sobre "cocina")
- Documento C: [0.023, 0.039, 0.002, 0.016, ...] (sobre "embeddings")
```

### 📊 Paso 3: Cálculo de Similaridad

```
Similaridad Coseno = mide el "ángulo" entre vectores

Consulta vs Documento A: 0.95 (muy similar)
Consulta vs Documento B: 0.12 (muy diferente)
Consulta vs Documento C: 0.89 (similar)

Ranking final: A (0.95) > C (0.89) > B (0.12)
```

### 🌐 **URL para demostrar:**

```
http://localhost:8080/manual/search?q=bases%20de%20datos%20vectoriales&backend=qdrant&k=3
```

---

## 🎓 **PARTE 3: Demostración Completa**

### 🌐 **URL principal para la clase:**

```
http://localhost:8080/manual/demo?q=bases%20de%20datos%20vectoriales&backend=qdrant
```

Esta URL muestra **todo el proceso junto** en una sola página optimizada para zoom y proyección.

---

## 🔧 **PARTE 4: Filtros de Metadata**

### 🎯 ¿Por qué usar filtros?

A veces queremos buscar solo en:

- Documentos PDF (no archivos de texto)
- Sección específica (solo "objetivos")
- Páginas específicas
- Documentos que contengan ciertas palabras

### 📋 Ejemplos de Filtros Disponibles

#### **1. Por tipo de documento**

```
Solo PDFs: /ask?q=vectores&document_type=pdf
Solo archivos texto: /ask?q=vectores&document_type=txt
```

#### **2. Por sección del curso**

```
Solo objetivos: /ask?q=evaluacion&section=objetivos
Solo cronograma: /ask?q=fechas&section=cronograma
Solo evaluación: /ask?q=proyecto&section=evaluacion
```

#### **3. Por tema específico**

```
Solo bases vectoriales: /ask?q=busqueda&topic=vectorial
Solo NoSQL: /ask?q=mongodb&topic=nosql
```

#### **4. Por página (PDFs)**

```
Solo página 5: /ask?q=proyecto&page=5
```

#### **5. Debe contener palabra**

```
Debe mencionar "NoSQL": /ask?q=bases&contains=NoSQL
```

#### **6. Combinación de filtros**

```
Objetivos en PDFs sobre vectores:
/ask?q=vectoriales&document_type=pdf&section=objetivos&topic=vectorial
```

### 🌐 **URL para ver todos los filtros:**

```
http://localhost:8080/filters/examples
```

---

## 🎯 **PARTE 5: Comparación de Motores**

### 🔧 Qdrant vs PostgreSQL+pgvector

#### **Qdrant**

- ✅ Especializado en vectores
- ✅ Búsqueda muy rápida (algoritmo HNSW)
- ✅ Filtros avanzados nativos
- ✅ Escalabilidad masiva

#### **PostgreSQL + pgvector**

- ✅ Integración con datos relacionales
- ✅ ACID transactions
- ✅ SQL familiar
- ⚠️ Menos optimizado para vectores puros

### 🌐 **URL para comparar:**

```
http://localhost:8080/compare?q=bases%20de%20datos%20vectoriales&k=3
```

---

## 📝 **SECUENCIA RECOMENDADA PARA LA CLASE**

### **1. Introducción (5 min)**

- Explicar problema: búsqueda por palabras exactas vs. significado
- Mostrar URL principal: `http://localhost:8080/`

### **2. Vectorización Manual (10 min)**

- URL: `http://localhost:8080/manual/embed?q=bases%20de%20datos%20vectoriales`
- Explicar cada paso
- Mostrar cómo el texto se convierte en números

### **3. Búsqueda Manual (10 min)**

- URL: `http://localhost:8080/manual/search?q=bases%20de%20datos%20vectoriales&backend=qdrant&k=3`
- Mostrar comparación de vectores
- Explicar similaridad coseno

### **4. Demo Completa (5 min)**

- URL: `http://localhost:8080/manual/demo?q=bases%20de%20datos%20vectoriales&backend=qdrant`
- Resumen de todo el proceso

### **5. Filtros Prácticos (10 min)**

- URL: `http://localhost:8080/filters/examples`
- Mostrar casos de uso reales
- Probar algunos filtros en vivo

### **6. Comparación de Motores (5 min)**

- URL: `http://localhost:8080/compare?q=bases%20de%20datos%20vectoriales&k=3`
- Mostrar diferencias entre Qdrant y PostgreSQL

### **7. Pruebas en Vivo (10 min)**

- Dejar que estudiantes sugieran consultas
- Probar con: `/ask?q=[consulta_estudiante]&backend=qdrant&k=3`

---

## 🚀 **URLs Rápidas para Copy-Paste**

```bash
# Página principal
http://localhost:8080/

# Demo completa (MÁS IMPORTANTE)
http://localhost:8080/manual/demo?q=bases%20de%20datos%20vectoriales&backend=qdrant

# Proceso de vectorización
http://localhost:8080/manual/embed?q=bases%20de%20datos%20vectoriales

# Proceso de búsqueda
http://localhost:8080/manual/search?q=bases%20de%20datos%20vectoriales&backend=qdrant&k=3

# Ejemplos de filtros
http://localhost:8080/filters/examples

# Comparación de motores
http://localhost:8080/compare?q=bases%20de%20datos%20vectoriales&k=3

# Búsqueda simple
http://localhost:8080/ask?q=vectores&backend=qdrant&k=3

# Búsqueda con filtros
http://localhost:8080/ask?q=vectores&backend=qdrant&k=3&section=objetivos&document_type=pdf
```

---

## 🎓 **Conceptos Clave para Enfatizar**

### **1. Semántica vs. Léxica**

- ❌ Búsqueda tradicional: palabras exactas
- ✅ Búsqueda vectorial: significado semántico

### **2. Por qué Funciona**

- Textos similares → vectores similares
- Entrenamiento masivo en múltiples idiomas
- Captura relaciones complejas

### **3. Ventajas Reales**

- Funciona en español
- No necesita palabras exactas
- Encuentra sinónimos automáticamente
- Escalable a millones de documentos

### **4. Aplicaciones Prácticas**

- Sistemas de recomendación
- Búsqueda en documentos
- Análisis de sentimientos
- Traducción automática
- Chatbots inteligentes

---

## 🔧 **Preparación Técnica**

### **Antes de la Clase:**

```bash
# 1. Iniciar servicios
docker compose up -d

# 2. Verificar que la API funciona
curl http://localhost:8080/

# 3. Probar demo principal
curl "http://localhost:8080/manual/demo?q=bases%20de%20datos%20vectoriales"
```

### **Durante la Clase:**

- Tener las URLs copiadas y listas
- Navegador en modo pantalla completa
- Zoom al 150% para mejor visibilidad
- Tener consultas de ejemplo preparadas

---

## 📚 **Recursos Adicionales**

### **Para Estudiantes Avanzados:**

- Documentación del modelo E5: [https://huggingface.co/intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
- Paper original de similaridad coseno
- Arquitectura HNSW de Qdrant

### **Para Desarrollo:**

- Código fuente en: `/app/main.py`
- Endpoints manuales en: `/manual/*`
- Lógica de filtros en: `/app/*_backend.py`

---

> **💡 Tip para el Profesor:** Las URLs están diseñadas para ser **browser-friendly** y mostrar información clara en formato JSON. Usa Ctrl+Plus para hacer zoom y que los estudiantes vean mejor desde atrás del aula.
