def render_embedding_html(result, text):
    """Render HTML for embedding demo"""
    from app.main import get_svg_icon

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Embedding Demo: {text}</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            background: #1e1e1e; 
            color: #e1e1e1; 
            margin: 20px; 
            line-height: 1.6;
        }}
        .container {{ 
            max-width: 1000px; 
            margin: 0 auto; 
            background: #2a2a2a; 
            padding: 30px; 
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .title {{ 
            color: #4CAF50; 
            font-size: 28px; 
            margin-bottom: 20px; 
            text-align: center;
        }}
        .input-section {{ 
            background: #333; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px;
            border-left: 4px solid #4CAF50;
        }}
        .output-section {{ 
            background: #2d3748; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px;
            border-left: 4px solid #3182ce;
        }}
        .embedding-preview {{ 
            background: #1a1a1a; 
            padding: 15px; 
            border-radius: 5px; 
            font-family: 'Courier New', monospace;
            font-size: 12px;
            overflow-x: auto;
            border: 1px solid #444;
        }}
        .explanation {{ 
            background: #2d5a87; 
            padding: 15px; 
            border-radius: 8px; 
            margin-top: 20px;
            border-left: 4px solid #60a5fa;
        }}
        .stats {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
            margin: 20px 0;
        }}
        .stat {{ 
            background: #374151; 
            padding: 15px; 
            border-radius: 8px; 
            text-align: center;
        }}
        .stat-value {{ 
            font-size: 24px; 
            font-weight: bold; 
            color: #10b981;
        }}
        .stat-label {{ 
            color: #9ca3af; 
            font-size: 14px;
        }}
        .nav-links {{
            text-align: center; 
            margin-top: 30px;
        }}
        .nav-link {{
            color: #4CAF50; 
            text-decoration: none; 
            padding: 10px 20px; 
            border: 1px solid #4CAF50; 
            border-radius: 5px;
            margin: 0 10px;
            display: inline-block;
        }}
        .nav-link:hover {{
            background: #4CAF50;
            color: #1e1e1e;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="title">{get_svg_icon("brain", "32", "#4CAF50")} Demo de Embedding</div>
        
        <div class="input-section">
            <h3>{get_svg_icon("input", "20", "#4CAF50")} Texto de Entrada</h3>
            <p style="font-size: 18px; margin: 10px 0;">"{text}"</p>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{result['embedding_dimensions']}</div>
                <div class="stat-label">Dimensiones</div>
            </div>
            <div class="stat">
                <div class="stat-value">{result['embedding_model'].split('/')[-1]}</div>
                <div class="stat-label">Modelo</div>
            </div>
        </div>
        
        <div class="output-section">
            <h3>{get_svg_icon("output", "20", "#3182ce")} Vector Generado (Primeros 10 valores)</h3>
            <div class="embedding-preview">
                {result['embedding_preview'] if result['embedding_preview'] else 'No embedding generado'}
            </div>
        </div>
        
        <div class="explanation">
            <h3>{get_svg_icon("idea", "20", "#60a5fa")} Explicación</h3>
            <p>{result['explanation']}</p>
        </div>
        
        <div class="nav-links">
            <a href="/demo/pipeline?q={text}" class="nav-link">
                {get_svg_icon("experiment", "16", "#4CAF50")} Ver Demo Completo del Pipeline
            </a>
            <a href="/" class="nav-link">
                {get_svg_icon("home", "16", "#4CAF50")} Volver al Inicio
            </a>
        </div>
    </div>
</body>
</html>
"""
