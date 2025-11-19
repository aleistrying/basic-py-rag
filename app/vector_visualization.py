"""
Vector Space Visualization Module
Provides meaningful visualizations of document embeddings in 2D/3D space
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def reduce_dimensions(embeddings: np.ndarray, method: str = "umap", n_components: int = 2) -> np.ndarray:
    """
    Reduce high-dimensional embeddings to 2D/3D for visualization

    Args:
        embeddings: Array of shape (n_samples, n_features)
        method: "umap" or "tsne"
        n_components: 2 for 2D, 3 for 3D

    Returns:
        Reduced embeddings of shape (n_samples, n_components)
    """
    try:
        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42
                )
                return reducer.fit_transform(embeddings)
            except ImportError:
                logger.warning("UMAP not available, falling back to t-SNE")
                method = "tsne"

        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(30, len(embeddings) - 1),
                random_state=42,
                metric='cosine'
            )
            return reducer.fit_transform(embeddings)

    except Exception as e:
        logger.error(f"Dimension reduction failed: {e}")
        # Fallback: use PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(embeddings)


def get_available_collections() -> List[str]:
    """Get list of available collections from Qdrant"""
    from qdrant_client import QdrantClient
    try:
        client = QdrantClient(host="qdrant", port=6333)
        collections = client.get_collections()
        return [col.name for col in collections.collections]
    except Exception as e:
        logger.error(f"Error fetching collections: {e}")
        return ["course_docs_clean_cosine_hnsw"]


def fetch_embeddings_from_backend(
    backend: str = "qdrant",
    collection_name: str = "course_docs_clean_cosine_hnsw",
    limit: int = 5000,
    document_filter: Optional[str] = None
) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """
    Fetch embeddings and metadata from vector database backend (Qdrant or PgVector)

    Args:
        backend: "qdrant" or "pgvector"
        collection_name: Name of the collection (Qdrant only)
        limit: Maximum number of points to fetch
        document_filter: Optional document name to filter by

    Returns:
        Tuple of (embeddings_array, metadata_list, unique_documents)
    """
    if backend == "qdrant":
        from qdrant_client import QdrantClient

        try:
            client = QdrantClient(host="qdrant", port=6333)

            # Scroll through collection to get points
            points, _ = client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_vectors=True,
                with_payload=True
            )

            if not points:
                logger.warning(
                    f"No points found in collection {collection_name}")
                return np.array([]), [], []

            embeddings = []
            metadata = []
            unique_docs = set()

            for point in points:
                # Extract document name from metadata.source_name
                metadata_dict = point.payload.get("metadata", {})
                source_name = metadata_dict.get("source_name", "")
                doc_name = point.payload.get("document", "Unknown")

                # Use source_name if available, clean it up
                if source_name:
                    # Remove file extensions and clean up
                    doc_name = source_name.replace(".pdf", "").replace(
                        ".txt", "").replace(".yaml", "")
                    # Truncate long names
                    if len(doc_name) > 80:
                        doc_name = doc_name[:77] + "..."
                elif doc_name == "Unknown":
                    # Fallback to extracting from source_path
                    source_path = point.payload.get("source_path", "")
                    if source_path:
                        # Extract filename from path
                        doc_name = source_path.split(
                            "/")[-1].replace(".pdf", "").replace(".txt", "").replace(".yaml", "")
                        if len(doc_name) > 80:
                            doc_name = doc_name[:77] + "..."
                    else:
                        doc_name = f"Documento {point.id}"

                unique_docs.add(doc_name)

                # Apply document filter if specified
                if document_filter and document_filter != "all" and doc_name != document_filter:
                    continue

                embeddings.append(point.vector)
                metadata.append({
                    "id": point.id,
                    "document": doc_name,
                    "page": point.payload.get("page", "?"),
                    "content": point.payload.get("content", ""),
                    "chapter": point.payload.get("chapter"),
                    "section": point.payload.get("section")
                })

            return np.array(embeddings), metadata, sorted(list(unique_docs))

        except Exception as e:
            logger.error(f"Error fetching from Qdrant: {e}")
            return np.array([]), [], []

    elif backend == "pgvector":
        # PgVector doesn't store vectors directly accessible, would need to query all
        # For now, return empty - this would require significant refactoring
        logger.warning(
            "PgVector visualization not yet supported - vectors not directly accessible")
        return np.array([]), [], []

    else:
        logger.error(f"Unknown backend: {backend}")
        return np.array([]), [], []


# Keep backward compatibility
def fetch_embeddings_from_qdrant(
    collection_name: str = "course_docs_clean_cosine_hnsw",
    limit: int = 5000,
    document_filter: Optional[str] = None
) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """Backward compatibility wrapper"""
    return fetch_embeddings_from_backend("qdrant", collection_name, limit, document_filter)


def find_cluster_keywords(coords: np.ndarray, metadata: List[Dict], radius: float = 0.5, top_n: int = 3) -> List[Dict]:
    """
    Identify clusters and extract common keywords from nearby chunks

    Args:
        coords: 2D coordinates of points
        metadata: Metadata for each point
        radius: Distance threshold for clustering
        top_n: Number of top keywords to extract per cluster

    Returns:
        List of dictionaries with cluster info and keywords for each point
    """
    from collections import Counter
    import re

    cluster_info = []

    for i, coord in enumerate(coords):
        # Find nearby points
        distances = np.linalg.norm(coords - coord, axis=1)
        nearby_indices = np.where(distances < radius)[0]

        if len(nearby_indices) <= 1:
            cluster_info.append({"cluster_size": 1, "keywords": []})
            continue

        # Extract text from nearby chunks
        nearby_texts = [metadata[j]["content"]
                        for j in nearby_indices if j < len(metadata)]
        combined_text = " ".join(nearby_texts).lower()

        # Extract words (remove stopwords and short words)
        stopwords = {"de", "la", "el", "en", "y", "a", "los", "las", "un", "una", "por", "para",
                     "con", "del", "al", "lo", "que", "se", "es", "su", "o", "como", "the", "and",
                     "of", "to", "in", "a", "is", "for", "on", "with", "as", "page", "página"}
        words = re.findall(r'\b[a-záéíóúñ]{4,}\b', combined_text)
        words = [w for w in words if w not in stopwords]

        # Count and get top words
        word_counts = Counter(words)
        top_words = [word for word, _ in word_counts.most_common(top_n)]

        cluster_info.append({
            "cluster_size": len(nearby_indices),
            "keywords": top_words
        })

    return cluster_info


def create_visualization_data(
    embeddings: np.ndarray,
    metadata: List[Dict],
    query_embedding: Optional[np.ndarray] = None,
    query_text: Optional[str] = None,
    method: str = "umap"
) -> Dict:
    """
    Create visualization data with reduced dimensions

    Args:
        embeddings: Document embeddings
        metadata: Document metadata
        query_embedding: Optional query embedding to plot
        query_text: Text of the query
        method: Dimensionality reduction method

    Returns:
        Dictionary with visualization data ready for plotting
    """
    if len(embeddings) == 0:
        return {"error": "No embeddings available"}

    # Combine query with documents if provided
    if query_embedding is not None:
        all_embeddings = np.vstack(
            [embeddings, query_embedding.reshape(1, -1)])
        has_query = True
    else:
        all_embeddings = embeddings
        has_query = False

    # Reduce dimensions
    logger.info(
        f"Reducing {len(all_embeddings)} embeddings from {all_embeddings.shape[1]}D to 2D using {method}")
    coords_2d = reduce_dimensions(
        all_embeddings, method=method, n_components=2)

    # Separate query coordinates if present
    if has_query:
        doc_coords = coords_2d[:-1]
        query_coords = coords_2d[-1]
    else:
        doc_coords = coords_2d
        query_coords = None

    # Perform cluster analysis
    logger.info("Performing cluster analysis to find common keywords...")
    cluster_data = find_cluster_keywords(
        doc_coords, metadata, radius=1.0, top_n=3)

    # Group by document with cluster info
    documents = {}
    for i, meta in enumerate(metadata):
        doc_name = meta["document"]
        if doc_name not in documents:
            documents[doc_name] = {
                "name": doc_name,
                "points": [],
                "pages": [],
                "content_previews": []
            }

        cluster = cluster_data[i] if i < len(cluster_data) else {
            "cluster_size": 1, "keywords": []}

        documents[doc_name]["points"].append({
            "x": float(doc_coords[i][0]),
            "y": float(doc_coords[i][1]),
            "page": meta.get("page", "?"),
            "content": meta.get("content", ""),
            "cluster_size": cluster["cluster_size"],
            "keywords": cluster["keywords"]
        })
        documents[doc_name]["pages"].append(meta.get("page", "?"))
        documents[doc_name]["content_previews"].append(
            meta.get("content", "")[:50])

    # Prepare visualization data
    viz_data = {
        "method": method,
        "total_points": len(embeddings),
        "num_documents": len(documents),
        "documents": list(documents.values()),
        "query": None
    }

    if query_coords is not None and query_text:
        viz_data["query"] = {
            "text": query_text,
            "x": float(query_coords[0]),
            "y": float(query_coords[1])
        }

    return viz_data


def generate_scatter_plot_html(
    viz_data: Dict,
    title: str = "Vector Space Visualization",
    available_collections: List[str] = None,
    current_collection: str = "course_docs_clean_cosine_hnsw",
    available_documents: List[str] = None,
    current_document: str = "all",
    current_method: str = "umap"
) -> str:
    """
    Generate HTML with interactive Plotly scatter plot

    Args:
        viz_data: Visualization data from create_visualization_data
        title: Plot title
        available_collections: List of available collections for selector
        current_collection: Currently selected collection
        available_documents: List of available documents for filter
        current_document: Currently selected document filter
        current_method: Currently selected dimensionality reduction method

    Returns:
        HTML string with embedded Plotly visualization
    """
    import json

    if available_collections is None:
        available_collections = [current_collection]
    if available_documents is None:
        available_documents = []

    # Prepare data for Plotly with expanded color palette
    traces = []
    colors = [
        '#0ea5e9', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444',
        '#06b6d4', '#a855f7', '#14b8a6', '#f97316', '#ec4899',
        '#3b82f6', '#6366f1', '#22c55e', '#eab308', '#f43f5e',
        '#0891b2', '#9333ea', '#059669', '#ea580c', '#db2777',
        '#2563eb', '#7c3aed', '#16a34a', '#d97706', '#dc2626'
    ]

    for i, doc in enumerate(viz_data["documents"]):
        color = colors[i % len(colors)]

        x_coords = [p["x"] for p in doc["points"]]
        y_coords = [p["y"] for p in doc["points"]]
        pages = [p.get("page", "?") for p in doc["points"]]
        contents = [p.get("content", "")[:100] + "..." if len(p.get("content", ""))
                    > 100 else p.get("content", "") for p in doc["points"]]
        cluster_sizes = [p.get("cluster_size", 1) for p in doc["points"]]
        keywords = [p.get("keywords", []) for p in doc["points"]]

        hover_text = []
        for page, content, cluster_size, kws in zip(pages, contents, cluster_sizes, keywords):
            hover = f"<b>📄 {doc['name']}</b><br>"
            hover += f"<b>Página:</b> {page}<br>"
            if cluster_size > 1:
                hover += f"<b>🔍 Cluster:</b> {cluster_size} chunks cercanos<br>"
            if kws:
                hover += f"<b>🏷️ Temas comunes:</b> {', '.join(kws)}<br>"
            hover += f"<b>Contenido:</b> {content}"
            hover_text.append(hover)

        traces.append({
            "x": x_coords,
            "y": y_coords,
            "mode": "markers",
            "type": "scatter",
            "name": doc["name"][:40] + "..." if len(doc["name"]) > 40 else doc["name"],
            "marker": {
                "size": 8,
                "color": color,
                "opacity": 0.7,
                "line": {"width": 0.5, "color": "white"}
            },
            "text": hover_text,
            "hoverinfo": "text"
        })

    # Add query point if present
    if viz_data.get("query"):
        query = viz_data["query"]
        traces.append({
            "x": [query["x"]],
            "y": [query["y"]],
            "mode": "markers+text",
            "type": "scatter",
            "name": "Búsqueda",
            "marker": {
                "size": 15,
                "color": "#fbbf24",
                "symbol": "star",
                "line": {"width": 2, "color": "#ffffff"}
            },
            "text": ["🔍"],
            "textposition": "top center",
            "textfont": {"size": 20},
            "hovertext": f"<b>Búsqueda:</b><br>{query['text'][:100]}",
            "hoverinfo": "text",
            "showlegend": True
        })

    layout = {
        "title": {
            "text": title,
            "font": {"size": 24, "color": "#e5e7eb"}
        },
        "xaxis": {
            "title": f"Dimensión 1 ({viz_data['method'].upper()})",
            "showgrid": True,
            "gridcolor": "#374151",
            "color": "#9ca3af"
        },
        "yaxis": {
            "title": f"Dimensión 2 ({viz_data['method'].upper()})",
            "showgrid": True,
            "gridcolor": "#374151",
            "color": "#9ca3af"
        },
        "plot_bgcolor": "#1f2937",
        "paper_bgcolor": "#111827",
        "font": {"color": "#e5e7eb"},
        "hovermode": "closest",
        "legend": {
            "bgcolor": "#1f2937",
            "bordercolor": "#374151",
            "borderwidth": 1
        },
        "height": 700
    }

    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "vector_space_visualization",
            "height": 1000,
            "width": 1400,
            "scale": 2
        }
    }

    # Build collection options HTML
    collection_options = "\n".join([
        f'<option value="{col}" {"selected" if col == current_collection else ""}>{col}</option>'
        for col in available_collections
    ])

    # Build document filter options HTML
    doc_options = '<option value="all" selected>Todos los documentos</option>\n'
    doc_options += "\n".join([
        f'<option value="{doc}" {"selected" if doc == current_document else ""}>{doc[:60] + "..." if len(doc) > 60 else doc}</option>'
        for doc in available_documents
    ])

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{title}</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                background: #111827;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                color: #e5e7eb;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            .nav-bar {{
                background: linear-gradient(135deg, #4c1d95 0%, #8b5cf6 100%);
                padding: 15px 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            }}
            .nav-title {{
                font-size: 1.25rem;
                font-weight: 600;
                color: #ffffff;
            }}
            .nav-button {{
                background: rgba(255, 255, 255, 0.2);
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.875rem;
                text-decoration: none;
                transition: all 0.2s;
            }}
            .nav-button:hover {{
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-1px);
            }}
            .controls {{
                background: #1f2937;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                border: 1px solid #374151;
            }}
            .controls h3 {{
                margin: 0 0 15px 0;
                color: #8b5cf6;
                font-size: 1.1rem;
            }}
            .control-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
            }}
            .control-group {{
                display: flex;
                flex-direction: column;
            }}
            .control-label {{
                font-size: 0.875rem;
                color: #9ca3af;
                margin-bottom: 5px;
            }}
            .control-select {{
                background: #111827;
                color: #e5e7eb;
                border: 1px solid #374151;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 0.875rem;
                cursor: pointer;
            }}
            .control-select:focus {{
                outline: none;
                border-color: #8b5cf6;
            }}
            .apply-button {{
                background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                color: #ffffff;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.875rem;
                font-weight: 600;
                transition: all 0.2s;
                margin-top: auto;
            }}
            .apply-button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: #1f2937;
                border-radius: 10px;
                border-left: 4px solid #0ea5e9;
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                color: #0ea5e9;
                font-size: 2rem;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .stat-card {{
                background: #1f2937;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #374151;
            }}
            .stat-value {{
                font-size: 2rem;
                font-weight: 600;
                color: #0ea5e9;
            }}
            .stat-label {{
                font-size: 0.875rem;
                color: #9ca3af;
                margin-top: 5px;
            }}
            #plot {{
                background: #1f2937;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            }}
            .description {{
                margin-top: 20px;
                padding: 20px;
                background: #1f2937;
                border-radius: 10px;
                border-left: 4px solid #8b5cf6;
            }}
            .description h3 {{
                margin-top: 0;
                color: #8b5cf6;
            }}
            .description p {{
                line-height: 1.6;
                color: #d1d5db;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="nav-bar">
                <div class="nav-title">🎯 Visualización del Espacio Vectorial</div>
                <a href="/" class="nav-button">← Volver al Inicio</a>
            </div>
            
            <div class="controls">
                <h3>⚙️ Configuración</h3>
                <form method="get" action="/visualize/vectors">
                    <div class="control-grid">
                        <div class="control-group">
                            <label class="control-label">Colección</label>
                            <select name="collection" class="control-select" id="collectionSelect">
                                {collection_options}
                            </select>
                        </div>
                        <div class="control-group">
                            <label class="control-label">Documento</label>
                            <select name="document" class="control-select" id="documentSelect">
                                {doc_options}
                            </select>
                        </div>
                        <div class="control-group">
                            <label class="control-label">Método de Reducción</label>
                            <select name="method" class="control-select">
                                <option value="umap" {"selected" if current_method == "umap" else ""}>UMAP (Recomendado)</option>
                                <option value="tsne" {"selected" if current_method == "tsne" else ""}>t-SNE</option>
                            </select>
                        </div>
                        <div class="control-group">
                            <label class="control-label">Límite de Puntos</label>
                            <select name="limit" class="control-select">
                                <option value="500">500 puntos</option>
                                <option value="1000">1,000 puntos</option>
                                <option value="2000">2,000 puntos</option>
                                <option value="5000" selected>5,000 puntos (Todos)</option>
                            </select>
                        </div>
                        <div class="control-group">
                            <button type="submit" class="apply-button">Aplicar Filtros</button>
                        </div>
                    </div>
                </form>
            </div>
            
            <div class="header">
                <h1>Espacio Vectorial: {current_collection}</h1>
                <p>Representación 2D de embeddings usando {viz_data['method'].upper()}</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{viz_data['total_points']}</div>
                    <div class="stat-label">Chunks Totales</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{viz_data['num_documents']}</div>
                    <div class="stat-label">Documentos</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{viz_data['method'].upper()}</div>
                    <div class="stat-label">Método de Reducción</div>
                </div>
                {'<div class="stat-card"><div class="stat-value">🔍</div><div class="stat-label">Con Búsqueda</div></div>' if viz_data.get('query') else ''}
            </div>
            
            <div id="plot"></div>
            
            <div class="description">
                <h3>📊 Cómo Interpretar Esta Visualización</h3>
                <p><strong>Puntos cercanos = Contenido similar:</strong> Los chunks que aparecen cerca en este espacio 2D tienen embeddings similares, lo que significa que su contenido es semánticamente relacionado.</p>
                <p><strong>Clusters de colores:</strong> Cada color representa un documento diferente. Puedes ver cómo se distribuyen los chunks de cada libro/documento en el espacio vectorial.</p>
                <p><strong>Overlap entre documentos:</strong> Si chunks de diferentes documentos aparecen juntos, significa que esos documentos tratan temas similares en esas secciones.</p>
                {'<p><strong>Estrella amarilla (🔍):</strong> Representa tu búsqueda. Los chunks más cercanos a este punto son los más relevantes para tu consulta.</p>' if viz_data.get('query') else ''}
                <p><strong>Método {viz_data['method'].upper()}:</strong> {'UMAP preserva la estructura global y local de los datos, mostrando clusters naturales.' if viz_data['method'] == 'umap' else 't-SNE enfatiza la estructura local, agrupando puntos similares.'}</p>
            </div>
        </div>
        
        <script>
            var data = {json.dumps(traces)};
            var layout = {json.dumps(layout)};
            var config = {json.dumps(config)};
            
            Plotly.newPlot('plot', data, layout, config);
        </script>
    </body>
    </html>
    """

    return html
