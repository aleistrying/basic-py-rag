"""
Template rendering utilities and SVG icon generation for the RAG application.
"""

from markupsafe import Markup
from typing import Dict, Any
import re


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
        "clock": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12,6 12,12 16,14"/></svg>',
        "check": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polyline points="20,6 9,17 4,12"/></svg>',
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
        "layers": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polygon points="12,2 2,7 12,12 22,7"/><polyline points="2,17 12,22 22,17"/><polyline points="2,12 12,17 22,12"/></svg>',
        "brain": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M12 5C10.7 5 9 6.2 9 7.5c0-.9-.8-1.6-1.7-1.6C6.4 5.9 6 6.8 6 7.8 6 9.1 7.2 10 8.5 10H9c0 1.3 1.2 2.5 2.5 2.5h1c1.3 0 2.5-1.2 2.5-2.5h.5c1.3 0 2.5-.9 2.5-2.2 0-1-.4-1.9-1.3-1.9-.9 0-1.7.7-1.7 1.6C15 6.2 13.3 5 12 5z"/><path d="M12 19c1.3 0 3-1.2 3-2.5 0 .9.8 1.6 1.7 1.6.9 0 1.3-.9 1.3-1.9 0-1.3-1.2-2.2-2.5-2.2H15c0-1.3-1.2-2.5-2.5-2.5h-1c-1.3 0-2.5 1.2-2.5 2.5h-.5c-1.3 0-2.5.9-2.5 2.2 0 1 .4 1.9 1.3 1.9.9 0 1.7-.7 1.7-1.6C9 17.8 10.7 19 12 19z"/></svg>',
        "refresh": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M21 21v-5h-5"/></svg>',
        "document": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14,2 14,8 20,8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10,9 9,9 8,9"/></svg>',
        "puzzle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M19.439 7.85c-.049.322-.059.648-.026.975.039.39.003.804-.115 1.198-.122.403-.31.787-.561 1.121-.255.345-.57.647-.928.885-.351.234-.748.403-1.15.493-.37.083-.751.072-1.114.025C14.904 12.674 14.28 13 13.6 13c-.68 0-1.304-.326-1.945-.189-.363.047-.744.058-1.114-.025-.402-.09-.799-.259-1.15-.493-.358-.238-.673-.54-.928-.885-.251-.334-.439-.718-.561-1.121-.118-.394-.154-.808-.115-1.198.033-.327.023-.653-.026-.975C7.694 9.486 7.3 9.1 7.3 8.6c0-.5.394-.886.461-1.25.049-.322.059-.648.026-.975-.039-.39-.003-.804.115-1.198.122-.403.31-.787.561-1.121C8.718 3.71 9.033 3.408 9.391 3.17c.351-.234.748-.403 1.15-.493.37-.083.751-.072 1.114-.025C12.296 2.526 12.92 2.2 13.6 2.2c.68 0 1.304.326 1.945.189.363-.047.744-.058 1.114.025.402.09.799.259 1.15.493.358.238.673.54.928.885.251.334.439.718.561 1.121.118.394.154.808.115 1.198-.033.327-.023.653.026.975C19.506 7.414 19.9 7.8 19.9 8.3c0 .5-.394.886-.461 1.25z"/></svg>',
        "shuffle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polyline points="16,3 21,3 21,8"/><line x1="4" y1="20" x2="21" y2="3"/><polyline points="21,16 21,21 16,21"/><line x1="15" y1="15" x2="21" y2="21"/><line x1="4" y1="4" x2="9" y2="9"/></svg>',
        "repeat": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polyline points="17,1 21,5 17,9"/><path d="M3 11v-1a4 4 0 0 1 4-4h14"/><polyline points="7,23 3,19 7,15"/><path d="M21 13v1a4 4 0 0 1-4 4H3"/></svg>',
        "fire": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M12 2c1.5 0 3 1 3 1s1.5-1 3-1c1.5 0 3 1 3 3 0 4-3 7-9 13-6-6-9-9-9-13 0-2 1.5-3 3-3z"/></svg>',
        "edit": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M20 14.66V20a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h5.34"/><polygon points="18,2 22,6 12,16 8,16 8,12"/></svg>',
        "scissors": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><line x1="20" y1="4" x2="8.12" y2="15.88"/><line x1="14.47" y1="14.48" x2="20" y2="20"/><line x1="8.12" y1="8.12" x2="12" y2="12"/></svg>',
        "clean": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M3 6h18l-2 13H5L3 6z"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>',
        "chart": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
        "trending-up": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polyline points="23,6 13.5,15.5 8.5,10.5 1,18"/><polyline points="17,6 23,6 23,12"/></svg>',
        "calculator": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><rect x="4" y="2" width="16" height="20" rx="2"/><line x1="8" y1="6" x2="16" y2="6"/><line x1="8" y1="10" x2="16" y2="10"/><line x1="8" y1="14" x2="16" y2="14"/><line x1="8" y1="18" x2="16" y2="18"/></svg>',
        "gear": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
        "upload": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17,8 12,3 7,8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
        "folder": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>',
        "trash": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polyline points="3,6 5,6 21,6"/><path d="m19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>',
        "shuffle-variant": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v10a4 4 0 0 1-4 4H2M22 3h-6a4 4 0 0 0-4 4v10a4 4 0 0 0 4 4h6M7 12h10"/><path d="M7 8l-4 4 4 4M17 8l4 4-4 4"/></svg>',
        "repeat-variant": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polyline points="7,8 3,12 7,16"/><line x1="21" y1="12" x2="3" y2="12"/><polyline points="17,16 21,12 17,8"/></svg>',
        "puzzle-piece": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M19.439 7.85c-.049.322-.059.648-.026.975.039.39.003.804-.115 1.198-.122.403-.31.787-.561 1.121-.255.345-.57.647-.928.885-.351.234-.748.403-1.15.493-.37.083-.751.072-1.114.025C14.904 12.674 14.28 13 13.6 13c-.68 0-1.304-.326-1.945-.189-.363.047-.744.058-1.114-.025-.402-.09-.799-.259-1.15-.493-.358-.238-.673-.54-.928-.885-.251-.334-.439-.718-.561-1.121-.118-.394-.154-.808-.115-1.198.033-.327.023-.653-.026-.975C7.694 9.486 7.3 9.1 7.3 8.6c0-.5.394-.886.461-1.25.049-.322.059-.648.026-.975-.039-.39-.003-.804.115-1.198.122-.403.31-.787.561-1.121C8.718 3.71 9.033 3.408 9.391 3.17c.351-.234.748-.403 1.15-.493.37-.083.751-.072 1.114-.025C12.296 2.526 12.92 2.2 13.6 2.2c.68 0 1.304.326 1.945.189.363-.047.744-.058 1.114.025.402.09.799.259 1.15.493.358.238.673.54.928.885.251.334.439.718.561 1.121.118.394.154.808.115 1.198-.033.327-.023.653.026.975C19.506 7.414 19.9 7.8 19.9 8.3c0 .5-.394.886-.461 1.25z"/></svg>',
        "hash": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/></svg>',
        "ruler": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M21.3 8.7l-9.6 9.6c-.4.4-1 .4-1.4 0L1.7 9.7c-.4-.4-.4-1 0-1.4L11.3.7c.4-.4 1-.4 1.4 0l8.6 8.6c.4.4.4 1 0 1.4z"/><path d="M7.5 10.5L9 12M10.5 7.5L12 9M13.5 4.5L15 6"/></svg>',
        "hard-drive": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><line x1="22" y1="12" x2="2" y2="12"/><path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/><line x1="6" y1="16" x2="6.01" y2="16"/><line x1="10" y1="16" x2="10.01" y2="16"/></svg>',
        "settings-gear": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
        "elephant": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M6 4c1.5-1.5 4-2 6.5-1.5S16 4 17 6c1 2 1 4 0 6s-3 3-5 3-4-1-5-3-1-4 0-6c0.5-1 1.5-1.5 2.5-1.5"/><path d="M14 7v1M10 7v1M12 11c-1 1-2 1-3 0"/></svg>',
        "question-circle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "abacus": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><rect x="2" y="3" width="20" height="18" rx="1"/><line x1="7" y1="7" x2="7" y2="17"/><line x1="12" y1="7" x2="12" y2="17"/><line x1="17" y1="7" x2="17" y2="17"/><circle cx="7" cy="10" r="1"/><circle cx="12" cy="8" r="1"/><circle cx="17" cy="12" r="1"/></svg>',
        "shuffle-double": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polyline points="16,3 21,3 21,8"/><line x1="4" y1="20" x2="21" y2="3"/><polyline points="21,16 21,21 16,21"/><line x1="15" y1="15" x2="21" y2="21"/><line x1="4" y1="4" x2="9" y2="9"/></svg>',
        "file-text": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14,2 14,8 20,8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10,9 9,9 8,9"/></svg>'
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


def convert_markdown_to_html(text: str) -> str:
    """Convert simple markdown formatting to HTML"""
    if not text:
        return ""

    # Convert **bold** to <strong>
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)

    # Convert *italic* to <em>
    text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)

    # Convert line breaks to <br>
    text = text.replace('\n', '<br>')

    return text


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
    env.filters['markdown_to_html'] = convert_markdown_to_html

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
        title="Demostración de Embedding",
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
        title="Demostración de Búsqueda",
        title_color="#10b981"
    )


def render_pretty_json(data: dict) -> str:
    """Render JSON data in a pretty format"""
    import json
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        return str(data)
