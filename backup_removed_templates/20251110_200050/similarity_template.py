def render_similarity_html(result):
    """Render HTML for similarity demo"""
    from app.main import get_svg_icon

    text1 = result['text1']
    text2 = result['text2']
    cosine_similarity = result['cosine_similarity']
    cosine_distance = result['cosine_distance']

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Similarity Demo</title>
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
        .text-pair {{ 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
            margin: 20px 0;
        }}
        .text-box {{ 
            background: #333; 
            padding: 20px; 
            border-radius: 8px; 
            border-left: 4px solid #4CAF50;
        }}
        .results {{ 
            background: #2d3748; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 20px 0;
            text-align: center;
        }}
        .calculation {{ 
            background: #1a1a1a; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 10px 0;
            font-family: 'Courier New', monospace;
        }}
        .score {{ 
            font-size: 32px; 
            font-weight: bold; 
            color: #10b981; 
            margin: 20px 0;
        }}
        .interpretation {{
            background: #2d5a87; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 20px 0;
            border-left: 4px solid #60a5fa;
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
        @media (max-width: 768px) {{
            .text-pair {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="title">{get_svg_icon("ruler", "32", "#4CAF50")} Demo de Similitud de Coseno</div>
        
        <div class="text-pair">
            <div class="text-box">
                <h3>{get_svg_icon("input", "20", "#4CAF50")} Texto 1</h3>
                <p>"{text1}"</p>
            </div>
            <div class="text-box">
                <h3>{get_svg_icon("input", "20", "#4CAF50")} Texto 2</h3>
                <p>"{text2}"</p>
            </div>
        </div>
        
        <div class="results">
            <h3>{get_svg_icon("calculator", "20", "#3182ce")} Similitud de Coseno</h3>
            <div class="score">{cosine_similarity:.6f}</div>
            <p>{'📈 Alta similitud' if cosine_similarity > 0.7 else '📊 Similitud media' if cosine_similarity > 0.4 else '📉 Baja similitud'}</p>
        </div>
        
        <div class="calculation">
            <strong>Similitud de Coseno:</strong> {cosine_similarity:.6f}
        </div>
        <div class="calculation">
            <strong>Distancia de Coseno:</strong> 1 - {cosine_similarity:.6f} = {cosine_distance:.6f}
        </div>
        
        <div class="interpretation">
            <h3>{get_svg_icon("idea", "20", "#60a5fa")} Interpretación</h3>
            <p>{result['interpretation']}</p>
            <ul>
                <li><strong>0.8-1.0:</strong> Muy similares</li>
                <li><strong>0.6-0.8:</strong> Similares</li>
                <li><strong>0.4-0.6:</strong> Algo relacionados</li>
                <li><strong>0.0-0.4:</strong> Diferentes</li>
            </ul>
        </div>
        
        <div class="nav-links">
            <a href="/demo/pipeline?q=comparación de similitud" class="nav-link">
                {get_svg_icon("experiment", "16", "#4CAF50")} Demo Completo del Pipeline
            </a>
            <a href="/" class="nav-link">
                {get_svg_icon("home", "16", "#4CAF50")} Volver al Inicio
            </a>
        </div>
    </div>
</body>
</html>
"""
