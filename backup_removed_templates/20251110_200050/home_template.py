"""
Home page template for RAG Demo with dynamic variables
"""


def render_home_page_html(data: dict) -> str:
    """HTML template for home page with search interface and quick actions"""

    # Default data if none provided
    if data is None:
        data = {
            "title": "RAG Demo",
            "subtitle": "Sistema de Búsqueda Inteligente con IA",
            "version": "3.0",
            "search_placeholder": "Ej: ¿Cuáles son las vacunas recomendadas para embarazadas?",
            "quick_searches": [
                {"label": "Vacunas", "query": "vacunas+embarazadas"},
                {"label": "Hipertensión", "query": "hipertensión+embarazo"},
                {"label": "Diabetes", "query": "diabetes+gestacional"},
                {"label": "Parto", "query": "parto+cesárea"},
                {"label": "Lactancia", "query": "lactancia+medicamentos"},
                {"label": "Ultrasonido", "query": "ultrasonido+embarazo"}
            ],
            "feature_cards": [
                {
                    "title": "Solo Búsqueda (/ask)",
                    "desc": "Búsqueda semántica sin respuesta de IA",
                    "url": "/ask?q=bases+de+datos+vectoriales",
                    "icon": "target"
                },
                {
                    "title": "Respuesta con IA (/ai)",
                    "desc": "Búsqueda + respuesta generada por IA",
                    "url": "/ai?q=qué+es+la+preeclampsia",
                    "icon": "robot"
                },
                {
                    "title": "Comparar Motores",
                    "desc": "Qdrant vs PostgreSQL+pgvector",
                    "url": "/compare?q=embarazo+diabetes",
                    "icon": "compare"
                },
                {
                    "title": "Documentación API",
                    "desc": "OpenAPI/Swagger docs completas",
                    "url": "/docs",
                    "icon": "book"
                }
            ],
            "api_groups": [
                {
                    "title": "Búsqueda y Consultas",
                    "routes": [
                        {"name": "/ask - Búsqueda Semántica",
                            "url": "/ask?q=bases+de+datos", "class": "search-btn"},
                        {"name": "/ai - Búsqueda + IA",
                            "url": "/ai?q=que+es+nosql", "class": "ai-btn"},
                        {"name": "/compare - Comparar Motores",
                            "url": "/compare?q=vectores", "class": "compare-btn"}
                    ]
                },
                {
                    "title": "Demos Educativas",
                    "routes": [
                        {"name": "/demo/pipeline - Pipeline Completo",
                            "url": "/demo/pipeline?q=pgvector", "class": "demo-btn"},
                        {"name": "/demo/embedding - Crear Embeddings",
                            "url": "/demo/embedding?text=PostgreSQL", "class": "embed-btn"},
                        {"name": "/demo/similarity - Calcular Similitud",
                            "url": "/demo/similarity?text1=pgvector&text2=vectorial", "class": "manual-btn"},
                        {"name": "/demo/pipeline - Demo Educativa Completa",
                            "url": "/demo/pipeline?query=bases+de+datos+vectoriales", "class": "demo-btn"},
                        {"name": "/manual/embed - Vectorización",
                            "url": "/manual/embed?q=ejemplo", "class": "embed-btn"},
                        {"name": "/manual/search - Búsqueda Manual",
                            "url": "/manual/search?q=ejemplo", "class": "manual-btn"}
                    ]
                },
                {
                    "title": "Documentación y Filtros",
                    "routes": [
                        {"name": "/filters/examples - Ejemplos",
                            "url": "/filters/examples", "class": "filter-btn"},
                        {"name": "/docs - Swagger/OpenAPI",
                            "url": "/docs", "class": "docs-btn"},
                        {"name": "JSON Response",
                            "url": "/?format=json", "class": "json-btn"}
                    ]
                }
            ],
            "nav_links": [
                {"name": "Búsquedas Académicas",
                    "url": "/ai?q=evaluación+del+curso", "primary": True},
                {"name": "Ejemplos de Filtros",
                    "url": "/filters/examples", "primary": False},
                {"name": "Demo Pipeline RAG",
                    "url": "/demo/pipeline?query=bases+de+datos+vectoriales", "primary": False}
            ]
        }

    # Extract data with defaults
    title = data.get("title", "RAG Demo")
    subtitle = data.get("subtitle", "Sistema de Búsqueda Inteligente con IA")
    search_placeholder = data.get(
        "search_placeholder", "Ej: ¿Cuáles son las vacunas recomendadas para embarazadas?")
    quick_searches = data.get("quick_searches", [])
    feature_cards = data.get("feature_cards", [])
    api_groups = data.get("api_groups", [])
    nav_links = data.get("nav_links", [])

    # Helper function for SVG icons (simplified)
    def get_svg_icon(icon_type, size="16", color="#ffffff"):
        icons = {
            "rocket": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><path d="M12 2L13.09 8.26L22 9L13.09 9.74L12 16L10.91 9.74L2 9L10.91 8.26L12 2Z"/></svg>',
            "search": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>',
            "robot": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="m12 7v4"/><path d="M8 16l0 0"/><path d="M16 16l0 0"/></svg>',
            "target": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
            "compare": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><path d="M3 12h18m-9-9l-9 9 9 9"/></svg>',
            "book": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
            "settings": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 1v6m0 6v6"/></svg>',
            "graduation": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>',
            "filter": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><polygon points="22,3 2,3 10,12.46 10,19 14,21 14,12.46"/></svg>',
            "experiment": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><path d="M9 2v6l-3 7c-.5 1 .5 2 1.5 2h9c1 0 2-1 1.5-2L15 8V2"/><path d="M9 2h6"/></svg>',
            "brain": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/></svg>',
            "ruler": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><path d="M21.3 8.7l-6-6a1 1 0 0 0-1.4 0l-12 12a1 1 0 0 0 0 1.4l6 6a1 1 0 0 0 1.4 0l12-12a1 1 0 0 0 0-1.4z"/></svg>',
            "clipboard": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/></svg>',
            "balance": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4"/><path d="M21 12c.552 0 1-.448 1-1s-.448-1-1-1-1 .448-1 1 .448 1 1 1z"/></svg>',
            "chart": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><path d="M3 3v18h18"/><path d="M7 12l4-4 4 4 4-4"/></svg>',
            "medical": f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.29 1.51 4.04 3 5.5l7 7Z"/></svg>'
        }
        return icons.get(icon_type, f'<svg width="{size}" height="{size}" fill="{color}" viewBox="0 0 24 24"><circle cx="12" cy="12" r="2"/></svg>')

    # Build quick search buttons
    quick_search_html = ""
    for search in quick_searches:
        quick_search_html += f'''
        <a href="/ai?q={search['query']}" class="quick-btn">{get_svg_icon("medical", "16")} {search['label']}</a>'''

    # Build feature cards
    feature_cards_html = ""
    for card in feature_cards:
        feature_cards_html += f'''
        <div class="feature-card action-card" onclick="window.location.href='{card['url']}'">
            <div class="feature-icon">{get_svg_icon(card['icon'], "32")}</div>
            <div class="feature-title">{card['title']}</div>
            <div class="feature-desc">{card['desc']}</div>
        </div>'''

    # Build API groups
    api_groups_html = ""
    for group in api_groups:
        routes_html = ""
        for route in group['routes']:
            icon = "clipboard" if "search" in route['class'] else "robot" if "ai" in route['class'] else "balance" if "compare" in route['class'] else "experiment" if "demo" in route[
                'class'] else "brain" if "embed" in route['class'] else "search" if "manual" in route['class'] else "clipboard" if "filter" in route['class'] else "book" if "docs" in route['class'] else "chart"
            routes_html += f'''
            <a href="{route['url']}" class="api-btn {route['class']}">{get_svg_icon(icon, "16", "#ffffff")} {route['name']}</a>'''

        api_groups_html += f'''
        <div class="api-group">
            <h4>{get_svg_icon("clipboard", "20", "#3b82f6")} {group['title']}</h4>
            {routes_html}
        </div>'''

    # Build navigation links
    nav_links_html = ""
    for link in nav_links:
        class_name = "nav-link primary" if link.get(
            'primary', False) else "nav-link"
        nav_links_html += f'''
        <a href="{link['url']}" class="{class_name}">{get_svg_icon("graduation", "16")} {link['name']}</a>'''

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title} - Main Menu</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 50%, #16213e 100%);
                color: #e1e1e1; 
                margin: 0;
                padding: 0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            
            /* Loading Animations */
            .loading-overlay {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                z-index: 9999;
                justify-content: center;
                align-items: center;
                flex-direction: column;
            }}
            
            .loading-overlay.active {{
                display: flex;
            }}
            
            .spinner {{
                width: 50px;
                height: 50px;
                border: 4px solid #374151;
                border-top: 4px solid #2563eb;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 20px;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            .loading-text {{
                color: #e5e7eb;
                font-size: 16px;
                margin-top: 10px;
                animation: pulse 2s infinite;
            }}
            
            @keyframes pulse {{
                0%, 100% {{ opacity: 0.7; }}
                50% {{ opacity: 1; }}
            }}
            
            /* Button loading states */
            .btn-loading {{
                position: relative;
                pointer-events: none;
                opacity: 0.7;
            }}
            
            .btn-loading::after {{
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 16px;
                height: 16px;
                margin: -8px 0 0 -8px;
                border: 2px solid transparent;
                border-top: 2px solid currentColor;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }}
            
            /* Form loading animation */
            .search-loading {{
                display: none;
                align-items: center;
                gap: 8px;
                margin-top: 10px;
                color: #2563eb;
                font-size: 14px;
            }}
            
            .search-loading.active {{
                display: flex;
            }}
            
            .dots {{
                display: inline-flex;
                gap: 2px;
            }}
            
            .dot {{
                width: 4px;
                height: 4px;
                background: #2563eb;
                border-radius: 50%;
                animation: dot-bounce 1.4s infinite ease-in-out;
            }}
            
            .dot:nth-child(1) {{ animation-delay: -0.32s; }}
            .dot:nth-child(2) {{ animation-delay: -0.16s; }}
            .dot:nth-child(3) {{ animation-delay: 0s; }}
            
            @keyframes dot-bounce {{
                0%, 80%, 100% {{ 
                    transform: scale(0);
                }} 
                40% {{ 
                    transform: scale(1);
                }}
            }}
            
            .container {{ 
                max-width: 900px; 
                margin: 20px;
                background: rgba(26, 26, 26, 0.95); 
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.5);
                overflow: hidden;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #2d5aa0 0%, #1e3a5f 100%);
                padding: 40px 30px;
                text-align: center;
                color: white;
            }}
            .title {{ 
                font-size: 36px; 
                margin: 0 0 10px 0;
                font-weight: 700;
                background: linear-gradient(45deg, #ffffff, #a8e6cf);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .subtitle {{ 
                font-size: 18px;
                opacity: 0.9;
                margin: 0;
            }}
            .search-section {{
                padding: 40px 30px;
                text-align: center;
            }}
            .search-title {{
                font-size: 24px;
                margin-bottom: 20px;
                color: #4a90e2;
                font-weight: 600;
            }}
            .search-bar {{
                display: flex;
                gap: 15px;
                max-width: 600px;
                margin: 0 auto 30px auto;
            }}
            .search-input {{
                flex: 1;
                padding: 15px 20px;
                border: 2px solid #374151;
                border-radius: 12px;
                background: #1f2937;
                color: #e5e7eb;
                font-size: 16px;
                transition: all 0.3s;
            }}
            .search-input:focus {{
                outline: none;
                border-color: #4a90e2;
                box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.2);
            }}
            .search-input::placeholder {{ color: #9ca3af; }}
            .search-btn {{
                padding: 15px 25px;
                border: none;
                border-radius: 12px;
                background: linear-gradient(45deg, #4a90e2, #357abd);
                color: white;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: all 0.3s;
                min-width: 120px;
            }}
            .search-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(74, 144, 226, 0.4);
            }}
            .quick-searches {{
                margin-top: 20px;
            }}
            .quick-title {{
                font-size: 16px;
                color: #9ca3af;
                margin-bottom: 15px;
            }}
            .quick-buttons {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                justify-content: center;
            }}
            .quick-btn {{
                padding: 8px 16px;
                background: rgba(74, 144, 226, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 20px;
                color: #60a5fa;
                text-decoration: none;
                font-size: 14px;
                transition: all 0.3s;
            }}
            .quick-btn:hover {{
                background: rgba(74, 144, 226, 0.2);
                transform: translateY(-1px);
            }}
            .features-section {{
                padding: 30px;
                background: #1f2937;
            }}
            .features-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .feature-card {{
                background: #374151;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #4b5563;
                transition: all 0.3s;
            }}
            .feature-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                border-color: #60a5fa;
            }}
            .feature-icon {{
                font-size: 24px;
                margin-bottom: 10px;
            }}
            .feature-title {{
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 8px;
                color: #e5e7eb;
            }}
            .feature-desc {{
                color: #9ca3af;
                font-size: 14px;
                line-height: 1.5;
            }}
            .action-card {{
                background: linear-gradient(45deg, #1e40af, #1e3a8a);
                cursor: pointer;
            }}
            .action-card:hover {{
                background: linear-gradient(45deg, #2563eb, #1e40af);
            }}
            .nav-section {{
                padding: 30px;
                background: #111827;
                text-align: center;
            }}
            .nav-buttons {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                justify-content: center;
            }}
            .nav-link {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 12px 20px;
                background: #374151;
                color: #e5e7eb;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 500;
                transition: all 0.3s;
                border: 1px solid #4b5563;
            }}
            .nav-link:hover {{
                background: #4b5563;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            .nav-link.primary {{
                background: linear-gradient(45deg, #059669, #047857);
                border-color: #10b981;
            }}
            .nav-link.primary:hover {{
                background: linear-gradient(45deg, #10b981, #059669);
            }}
            
            .api-routes-section {{
                padding: 30px;
                background: #0f172a;
                border-top: 1px solid #1e293b;
            }}
            .section-title {{
                text-align: center;
                font-size: 24px;
                color: #e1e7ef;
                margin-bottom: 30px;
                font-weight: 600;
            }}
            .api-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 25px;
                max-width: 1000px;
                margin: 0 auto;
            }}
            .api-group {{
                background: #1e293b;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #334155;
                display: flex;
                flex-direction: column;
            }}
            .api-group h4 {{
                color: #3b82f6;
                margin: 0 0 15px 0;
                font-size: 16px;
                border-bottom: 2px solid #3b82f6;
                padding-bottom: 8px;
            }}
            .api-btn {{
                display: block;
                padding: 12px 16px;
                margin-bottom: 8px;
                text-decoration: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.3s;
                border: 1px solid transparent;
            }}
            .api-btn:last-child {{ margin-bottom: 0; }}
            
            .search-btn {{ background: #059669; color: white; }}
            .search-btn:hover {{ background: #047857; transform: translateY(-1px); }}
            
            .ai-btn {{ background: #2563eb; color: white; }}
            .ai-btn:hover {{ background: #1d4ed8; transform: translateY(-1px); }}
            
            .compare-btn {{ background: #7c3aed; color: white; }}
            .compare-btn:hover {{ background: #6d28d9; transform: translateY(-1px); }}
            
            .demo-btn {{ background: #ea580c; color: white; }}
            .demo-btn:hover {{ background: #c2410c; transform: translateY(-1px); }}
            
            .embed-btn {{ background: #dc2626; color: white; }}
            .embed-btn:hover {{ background: #b91c1c; transform: translateY(-1px); }}
            
            .manual-btn {{ background: #0891b2; color: white; }}
            .manual-btn:hover {{ background: #0e7490; transform: translateY(-1px); }}
            
            .filter-btn {{ background: #16a34a; color: white; }}
            .filter-btn:hover {{ background: #15803d; transform: translateY(-1px); }}
            
            .docs-btn {{ background: #6366f1; color: white; }}
            .docs-btn:hover {{ background: #4f46e5; transform: translateY(-1px); }}
            
            .json-btn {{ background: #374151; color: #e5e7eb; }}
            .json-btn:hover {{ background: #4b5563; transform: translateY(-1px); }}
            
            @media (max-width: 768px) {{
                .container {{ margin: 10px; }}
                .header {{ padding: 30px 20px; }}
                .search-section {{ padding: 30px 20px; }}
                .search-bar {{ flex-direction: column; }}
                .quick-buttons {{ flex-direction: column; align-items: center; }}
                .nav-buttons {{ flex-direction: column; }}
                .features-grid {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <!-- Global loading overlay -->
        <div class="loading-overlay" id="loadingOverlay">
            <div class="spinner"></div>
            <div class="loading-text">Procesando consulta...</div>
        </div>
        
        <div class="container">
            <div class="header">
                <div class="title">{get_svg_icon("rocket", "32", "#ffffff")} {title}</div>
                <div class="subtitle">{subtitle}</div>
            </div>
            
            <div class="search-section">
                <div class="search-title">{get_svg_icon("search", "24")} Buscar en Documentos</div>
                <div class="search-bar">
                    <input type="text" class="search-input" placeholder="{search_placeholder}" id="searchInput">
                    <button class="search-btn" onclick="searchWithAI()" id="searchButton">{get_svg_icon("robot", "18")} AI Search</button>
                </div>
                
                <!-- Loading indicator -->
                <div class="search-loading" id="searchLoading">
                    <span>Buscando</span>
                    <div class="dots">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                </div>
                
                <div class="quick-searches">
                    <div class="quick-title">{get_svg_icon("target", "20")} Búsquedas Rápidas:</div>
                    <div class="quick-buttons">
                        {quick_search_html}
                    </div>
                </div>
            </div>
            
            <div class="features-section">
                <div class="features-grid">
                    {feature_cards_html}
                </div>
            </div>
            
            <div class="api-routes-section">
                <div class="section-title">{get_svg_icon("settings", "24", "#ffffff")} Todas las Rutas API</div>
                <div class="api-grid">
                    {api_groups_html}
                </div>
            </div>
            
            <div class="nav-section">
                <div class="nav-buttons">
                    {nav_links_html}
                </div>
            </div>
        </div>
        
        <script>
            function showLoading(message = 'Procesando consulta...') {{
                document.getElementById('loadingOverlay').classList.add('active');
                document.querySelector('.loading-text').textContent = message;
            }}
            
            function showSearchLoading() {{
                document.getElementById('searchLoading').classList.add('active');
                document.getElementById('searchButton').classList.add('btn-loading');
                document.getElementById('searchButton').disabled = true;
            }}
            
            function searchWithAI() {{
                const query = document.getElementById('searchInput').value;
                if (query.trim()) {{
                    showSearchLoading();
                    showLoading('Generando respuesta con IA...');
                    // Small delay to show animation before redirect
                    setTimeout(() => {{
                        window.location.href = `/ai?q=${{encodeURIComponent(query.trim())}}`;
                    }}, 300);
                }}
            }}
            
            function searchOnly() {{
                const query = document.getElementById('searchInput').value;
                if (query.trim()) {{
                    showSearchLoading();
                    showLoading('Buscando documentos...');
                    setTimeout(() => {{
                        window.location.href = `/ask?q=${{encodeURIComponent(query.trim())}}`;
                    }}, 300);
                }}
            }}
            
            // Add loading to feature cards
            function navigateWithLoading(url, message) {{
                showLoading(message);
                setTimeout(() => {{
                    window.location.href = url;
                }}, 300);
            }}
            
            // Allow Enter key to search
            document.getElementById('searchInput').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') {{
                    searchWithAI();
                }}
            }});
            
            // Add loading to quick search buttons
            document.addEventListener('DOMContentLoaded', function() {{
                // Add loading to quick search buttons
                const quickButtons = document.querySelectorAll('.quick-btn');
                quickButtons.forEach(btn => {{
                    btn.addEventListener('click', function(e) {{
                        e.preventDefault();
                        showLoading('Ejecutando búsqueda rápida...');
                        setTimeout(() => {{
                            window.location.href = this.href;
                        }}, 300);
                    }});
                }});
                
                // Add loading to feature cards
                const featureCards = document.querySelectorAll('.action-card');
                featureCards.forEach(card => {{
                    const originalOnclick = card.getAttribute('onclick');
                    if (originalOnclick) {{
                        card.removeAttribute('onclick');
                        card.addEventListener('click', function() {{
                            if (originalOnclick.includes('/ask')) {{
                                showLoading('Preparando búsqueda...');
                            }} else if (originalOnclick.includes('/ai')) {{
                                showLoading('Iniciando IA...');
                            }} else if (originalOnclick.includes('/compare')) {{
                                showLoading('Comparando backends...');
                            }}
                            setTimeout(() => {{
                                eval(originalOnclick);
                            }}, 300);
                        }});
                    }}
                }});
                
                // Example text cycling
                const examples = [
                    "¿Cuáles son las vacunas recomendadas para embarazadas?",
                    "¿Qué es la preeclampsia?",
                    "Tratamiento de diabetes gestacional",
                    "Complicaciones del parto prematuro",
                    "Medicamentos seguros en el embarazo"
                ];
                
                const input = document.getElementById('searchInput');
                let currentExample = 0;
                
                function cycleExamples() {{
                    input.placeholder = examples[currentExample];
                    currentExample = (currentExample + 1) % examples.length;
                }}
                
                setInterval(cycleExamples, 3000);
            }});
        </script>
    </body>
    </html>
    """
