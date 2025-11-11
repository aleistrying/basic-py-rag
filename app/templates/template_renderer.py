"""
Template rendering utilities and SVG icon generation for the RAG application.
"""

from markupsafe import Markup
from typing import Dict, Any


def get_svg_icon(name: str, size: str = "24", color: str = "#3b82f6") -> Markup:
    """
    Generate SVG icon markup for the given icon name.

    Args:
        name: Icon name (e.g., 'search', 'robot', 'home')
        size: SVG size in pixels (default: "24")
        color: SVG stroke color (default: "#3b82f6")

    Returns:
        Markup: Safe HTML string containing the SVG icon
    """
    icons = {
        "search": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>',
        "robot": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M12 2c2.21 0 4 1.79 4 4h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h3c0-2.21 1.79-4 4-4z"/><circle cx="9" cy="11" r="1"/><circle cx="15" cy="11" r="1"/><path d="m9 16h6"/></svg>',
        "home": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9,22 9,12 15,12 15,22"/></svg>',
        "book": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
        "book-open": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',
        "clipboard": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/><path d="m16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/></svg>',
        "settings": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="m12 1v6m0 6v6"/></svg>',
        "balance": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="m9 12 2 2 4-4"/><path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"/><path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"/><path d="m3 12h6m6 0h6"/></svg>',
        "download": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
        "eye": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
        "eye-off": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="m9.88 9.88a3 3 0 1 0 4.24 4.24"/><path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 11 8 11 8a13.16 13.16 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.526 13.526 0 0 0 1 12s4 8 11 8a9.74 9.74 0 0 0 5.39-1.61"/><line x1="2" y1="2" x2="22" y2="22"/></svg>',
        "help-circle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "info": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
        "alert-circle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r="0.01"/></svg>',
        "target": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "cpu": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
        "filter": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polygon points="22,3 2,3 10,12.46 10,19 14,21 14,12.46"/></svg>',
        "lightbulb": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M12 2a7 7 0 0 1 7 7c0 2.38-1.19 4.47-3 5.74V17a1 1 0 0 1-1 1H9a1 1 0 0 1-1-1v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 0 1 7-7z"/><line x1="12" y1="22" x2="12" y2="18"/></svg>',
        "zap": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polygon points="13,2 3,14 12,14 11,22 21,10 12,10"/></svg>',
        "database": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>',
        "layers": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polygon points="12,2 2,7 12,12 22,7"/><polyline points="2,17 12,22 22,17"/><polyline points="2,12 12,17 22,12"/></svg>'
    }

    icon_html = icons.get(
        name, f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg>')
    return Markup(icon_html)  # Mark as safe to prevent HTML escaping


def adjust_color(color: str) -> str:
    """Adjust hex color to be slightly darker"""
    try:
        hex_color = color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darkened = tuple(max(0, c - 30) for c in rgb)
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"
    except Exception:
        return "#1e3a5f"


def format_json_highlight(text: str) -> str:
    """Basic JSON syntax highlighting using CSS classes"""
    if not text:
        return ""

    # Simple regex-based highlighting
    import re

    # Highlight strings (green)
    text = re.sub(r'"([^"]*)"(?=\s*:)',
                  r'<span class="json-key">"\1"</span>', text)
    text = re.sub(r':\s*"([^"]*)"',
                  r': <span class="json-string">"\1"</span>', text)

    # Highlight numbers (blue)
    text = re.sub(r':\s*(\d+\.?\d*)',
                  r': <span class="json-number">\1</span>', text)

    # Highlight booleans (purple)
    text = re.sub(r':\s*(true|false|null)',
                  r': <span class="json-boolean">\1</span>', text)

    return text


def safe_format_dict(data: Dict[str, Any], max_length: int = 1000) -> str:
    """Safely format dictionary data for display"""
    try:
        import json
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        if len(formatted) > max_length:
            formatted = formatted[:max_length] + "..."
        return formatted
    except Exception:
        return str(data)[:max_length]


def get_status_badge_class(status: str) -> str:
    """Get CSS class for status badge based on status value"""
    status_lower = status.lower()

    if status_lower in ['active', 'running', 'online', 'available', 'connected']:
        return 'status-success'
    elif status_lower in ['inactive', 'stopped', 'offline', 'unavailable', 'disconnected']:
        return 'status-error'
    elif status_lower in ['loading', 'pending', 'starting', 'connecting']:
        return 'status-warning'
    else:
        return 'status-neutral'


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


# Template rendering functions
def render_home_page() -> str:
    """Render the home page with action cards"""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    import os

    template_dir = os.path.join(os.path.dirname(__file__))
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    env.globals['get_svg_icon'] = get_svg_icon

    template = env.get_template('home_page.html')
    return template.render()


def render_search_response(data: dict, query: str) -> str:
    """Render search response template"""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    import os
    import json

    template_dir = os.path.join(os.path.dirname(__file__))
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    env.globals['get_svg_icon'] = get_svg_icon
    env.filters['format_json_highlight'] = format_json_highlight

    # Prepare the data for template rendering
    template_data = {
        'page_title': f"Búsqueda: {query}",
        'query': query,
        'json_data': json.dumps(data, indent=2, ensure_ascii=False),
        **data  # Spread the original data fields
    }

    # Extract specific fields if they exist in the data
    if 'results' in data:
        template_data['results'] = data['results']
        template_data['total_results'] = len(data['results'])

    if 'search_time_ms' in data:
        template_data['search_time_ms'] = data['search_time_ms']

    if 'backend' in data:
        template_data['backend'] = data['backend']

    template = env.get_template('search_response.html')
    return template.render(template_data)


def render_ai_response(data: dict, query: str) -> str:
    """Render AI response template"""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    import os
    import json

    template_dir = os.path.join(os.path.dirname(__file__))
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    env.globals['get_svg_icon'] = get_svg_icon
    env.filters['format_json_highlight'] = format_json_highlight

    # Prepare the data for template rendering
    template_data = {
        'page_title': f"AI Response: {query}",
        'query': query,
        'json_data': json.dumps(data, indent=2, ensure_ascii=False),
        **data  # Spread the original data fields
    }

    # Extract specific fields if they exist in the data
    if 'ai_response' in data:
        template_data['ai_response'] = data['ai_response']
    elif 'response' in data:
        template_data['ai_response'] = data['response']
    elif 'answer' in data:
        template_data['ai_response'] = data['answer']
    else:
        template_data['ai_response'] = "No se pudo generar una respuesta AI."

    if 'sources' in data:
        template_data['sources'] = data['sources']
    elif 'results' in data:
        template_data['sources'] = data['results']

    if 'total_results' in data:
        template_data['total_results'] = data['total_results']
    elif 'sources' in template_data:
        template_data['total_results'] = len(template_data['sources'])
    elif 'results' in data:
        template_data['total_results'] = len(data['results'])
    else:
        template_data['total_results'] = 0

    if 'backend' in data:
        template_data['backend'] = data['backend']
    else:
        template_data['backend'] = "N/A"

    if 'model' in data:
        template_data['model'] = data['model']
    elif 'model_used' in data:
        template_data['model'] = data['model_used']
    else:
        template_data['model'] = "N/A"

    template = env.get_template('ai_response.html')
    return template.render(template_data)


def render_general_response(data: dict, title: str = "Sistema RAG", title_color: str = "#3b82f6") -> str:
    """Render general response template"""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    import os
    import json

    template_dir = os.path.join(os.path.dirname(__file__))
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    env.globals['get_svg_icon'] = get_svg_icon
    env.filters['format_json_highlight'] = format_json_highlight
    env.filters['get_status_badge_class'] = get_status_badge_class

    # Prepare the data for template rendering
    template_data = {
        'page_title': title,
        'title_color': title_color,
        'json_data': json.dumps(data, indent=2, ensure_ascii=False),
        **data  # Spread the original data fields
    }

    # Extract specific fields if they exist in the data
    if 'qdrant' in data and 'postgres' in data:
        template_data['qdrant'] = data['qdrant']
        template_data['postgres'] = data['postgres']

    if 'results' in data:
        template_data['results'] = data['results']
        template_data['total_results'] = len(data['results'])

    if 'search_time_ms' in data:
        template_data['search_time_ms'] = data['search_time_ms']

    if 'backend' in data:
        template_data['backend'] = data['backend']

    if 'error' in data:
        template_data['error'] = data['error']

    if 'status' in data:
        template_data['status'] = data['status']

    template = env.get_template('general_response.html')
    return template.render(template_data)


def render_manual_embedding(embedding_result: dict, text: str) -> str:
    """Render manual embedding demo template"""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    import os

    template_dir = os.path.join(os.path.dirname(__file__))
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    env.globals['get_svg_icon'] = get_svg_icon
    env.filters['format_json_highlight'] = format_json_highlight

    template = env.get_template('general_response.html')
    return template.render(
        result=embedding_result,
        title="🔧 Demostración de Embedding",
        title_color="#8b5cf6"
    )


def render_manual_search(search_result: dict, query: str) -> str:
    """Render manual search demo template"""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    import os

    template_dir = os.path.join(os.path.dirname(__file__))
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    env.globals['get_svg_icon'] = get_svg_icon
    env.filters['format_json_highlight'] = format_json_highlight

    template = env.get_template('general_response.html')
    return template.render(
        result=search_result,
        title="🔍 Demostración de Búsqueda",
        title_color="#10b981"
    )


def render_pretty_json(data: dict) -> str:
    """Render JSON data in a pretty format"""
    import json
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        return str(data)
