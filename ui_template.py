import os
import base64
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import gradio as gr


class ThemeConfig:
    """Centralized theme configuration with validation."""

    def __init__(self):
        # Default color palette
        self.primary_color = "#0F6CBD"
        self.accent_color = "#C4314B"
        self.success_color = "#2E7D32"
        self.bg1 = "#F0F7FF"
        self.bg2 = "#E8F0FA"
        self.bg3 = "#DDE7F8"
        self.font_family = (
            "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
            "Roboto, 'Helvetica Neue', Arial, sans-serif"
        )

        # Metadata
        self.project_name = "Heart Project"
        self.year = "2025"
        self.about = ""
        self.description = ""
        self.meta_items: List[Tuple[str, str]] = []

        # Cache for CSS
        self._css_cache: Optional[str] = None

    def update_colors(self, **kwargs) -> None:
        """Update color scheme with validation."""
        valid_keys = {'primary', 'accent', 'success', 'bg1', 'bg2', 'bg3'}
        for key, value in kwargs.items():
            if key not in valid_keys or value is None:
                continue
            if not self._is_valid_color(value):
                raise ValueError(f"Invalid color format for {key}: {value}")
            setattr(self, f"{key}_color" if not key.startswith('bg') else key, value)
        self._invalidate_cache()

    def update_font(self, font_family: str) -> None:
        """Update font family."""
        if font_family and isinstance(font_family, str):
            self.font_family = font_family
            self._invalidate_cache()

    def update_meta(self, project_name: Optional[str] = None,
                    year: Optional[str] = None,
                    about: Optional[str] = None,
                    description: Optional[str] = None,
                    meta_items: Optional[List[Tuple[str, str]]] = None) -> None:
        """Update metadata."""
        if project_name is not None:
            self.project_name = project_name
        if year is not None:
            self.year = year
        if about is not None:
            self.about = about
        if description is not None:
            self.description = description
        if meta_items is not None:
            self.meta_items = meta_items

    @staticmethod
    def _is_valid_color(color: str) -> bool:
        """Validate hex color format."""
        return isinstance(color, str) and (
            color.startswith('#') and len(color) in (4, 7, 9)
        )

    def _invalidate_cache(self) -> None:
        """Clear CSS cache when theme changes."""
        self._css_cache = None

    def get_css(self) -> str:
        """Get or generate CSS with caching."""
        if self._css_cache is None:
            self._css_cache = self._build_css()
        return self._css_cache

    def _build_css(self) -> str:
        """Build the complete CSS string."""
        return f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.gradio-container {{
    min-height: 100vh !important;
    width: 100vw !important;
    margin: 0 !important;
    padding: 0px !important;
    background: linear-gradient(135deg, {self.bg1} 0%, {self.bg2} 50%, {self.bg3} 100%);
    background-size: 600% 600%;
    animation: gradientBG 7s ease infinite;
}}

/* Global font setup */
body, .gradio-container, .gr-block, .gr-markdown, .gr-button, .gr-input,
.gr-dropdown, .gr-number, .gr-plot, .gr-dataframe, .gr-accordion, .gr-form,
.gr-textbox, .gr-html, table, th, td, label, h1, h2, h3, h4, h5, h6, p, span, div {{
    font-family: {self.font_family} !important;
}}

@keyframes gradientBG {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}

/* Minimize spacing and padding */
.content-wrap {{
    padding: 2px !important;
    margin: 0 !important;
}}

/* Reduce component spacing */
.gr-row {{
    gap: 5px !important;
    margin: 2px 0 !important;
}}

.gr-column {{
    gap: 4px !important;
    padding: 4px !important;
}}

/* Accordion optimization */
.gr-accordion {{
    margin: 4px 0 !important;
}}

.gr-accordion .gr-accordion-content {{
    padding: 2px !important;
}}

/* Form elements spacing */
.gr-form {{
    gap: 2px !important;
}}

/* Button styling */
.gr-button {{
    margin: 2px 0 !important;
}}

/* DataFrame optimization */
.gr-dataframe {{
    margin: 4px 0 !important;
}}

/* Remove horizontal scroll from data preview */
.gr-dataframe .wrap {{
    overflow-x: auto !important;
    max-width: 100% !important;
}}

/* Plot optimization */
.gr-plot {{
    margin: 4px 0 !important;
}}

/* Reduce markdown margins */
.gr-markdown {{
    margin: 2px 0 !important;
}}

/* Footer positioning */
.sticky-footer {{
    position: fixed;
    bottom: 0px;
    left: 0;
    width: 100%;
    background: {self.bg1};
    padding: 6px !important;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    z-index: 1000;
}}
"""


# Global theme instance
_theme = ThemeConfig()


def configure(project_name: Optional[str] = None,
              year: Optional[str] = None,
              about: Optional[str] = None,
              description: Optional[str] = None,
              colors: Optional[Dict[str, str]] = None,
              font_family: Optional[str] = None,
              meta_items: Optional[List[Tuple[str, str]]] = None) -> None:
    """
    One-call configuration for the entire theme.

    Args:
        project_name: Name of the project
        year: Project year
        about: About project
        description: Project description
        colors: Dict with keys: primary, accent, success, bg1, bg2, bg3
        font_family: CSS font family string
        meta_items: List of (label, value) tuples for metadata
    """
    if colors:
        _theme.update_colors(**colors)
    if font_family:
        _theme.update_font(font_family)
    _theme.update_meta(project_name, year, about, description, meta_items)


def get_custom_css() -> str:
    """Get the current custom CSS."""
    return _theme.get_css()


def _image_to_base64(image_path: str) -> str:
    """
    Convert image to base64 string with better error handling.

    Args:
        image_path: Relative path to image file

    Returns:
        Base64 encoded string

    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    current_dir = Path(__file__).parent
    full_path = current_dir / image_path

    if not full_path.exists():
        raise FileNotFoundError(f"Image not found: {full_path}")

    with open(full_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_header(logo_path: str = "static/heart_sentinel.png") -> None:
    """
    Create a header with logo and project name.

    Args:
        logo_path: Path to logo image
    """
    with gr.Row():
        with gr.Column(scale=2):
            try:
                logo_base64 = _image_to_base64(logo_path)
                gr.HTML(
                    f"""<img src="data:image/png;base64,{logo_base64}"
                            alt="Logo"
                            style="height:120px;width:auto;margin:0 auto;margin-bottom:16px;display:block;">"""
                )
            except FileNotFoundError:
                gr.HTML("<div style='text-align:center;color:#999;'>Logo not found</div>")

        with gr.Column(scale=2):
            gr.HTML(f"""
<div style="display:flex;justify-content:flex-start;align-items:center;gap:30px;">
    <div>
        <h1 style="margin-bottom:0;color:{_theme.primary_color};font-size:2.5em;font-weight:bold;">
            {_theme.project_name}
        </h1>
        <p style="margin-top:4px;font-size:1.1em;color:#555;">{_theme.about}</p>
    </div>
</div>
""")


def create_footer(logo_path: str = "static/intelligent_retail.png",
                  creator_name: str = "Thi-Diem-My Le",
                  creator_link: str = "https://beacons.ai/elizabethmyn",
                  org_name: str = "AI VIET NAM",
                  org_link: str = "https://aivietnam.edu.vn/") -> gr.HTML:
    """
    Create a sticky footer with creator information.

    Args:
        logo_path: Path to logo image
        creator_name: Name of creator
        creator_link: Link to creator profile
        org_name: Organization name
        org_link: Link to organization

    Returns:
        Gradio HTML component
    """
    try:
        logo_base64 = _image_to_base64(logo_path)
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height:0px;width:auto;">'
    except FileNotFoundError:
        logo_html = ""

    footer_html = f"""
<style>
  .sticky-footer{{
    position:fixed;
    bottom:0px;
    left:0;
    width:100%;
    background:#E8F5E8;
    padding:10px;
    box-shadow:0 -2px 10px rgba(0,0,0,0.1);
    z-index:1000;
  }}
  .content-wrap{{padding-bottom:60px;}}
</style>
<div class="sticky-footer">
  <div style="text-align:center;font-size:18px;color:#888">
    Created by
    <a href="{creator_link}" target="_blank"
       style="color:#465C88;text-decoration:none;font-weight:bold;display:inline-flex;align-items:center;">
      {creator_name}
      {logo_html}
    </a>
    from
    <a href="{org_link}" target="_blank"
       style="color:#355724;text-decoration:none;font-weight:bold;">
      {org_name}
    </a>
  </div>
</div>
"""
    return gr.HTML(footer_html)


def render_info_card(description: Optional[str] = None,
                     meta_items: Optional[List[Tuple[str, str]]] = None,
                     icon: str = "🧠",
                     title: str = "About this demo") -> str:
    """
    Render an informational card.

    Args:
        description: Card description text
        meta_items: List of (label, value) tuples
        icon: Emoji or icon for the card
        title: Card title

    Returns:
        HTML string for the card
    """
    desc = description if description is not None else _theme.description
    items = meta_items if meta_items is not None else _theme.meta_items

    meta_html = ""
    if items:
        meta_html = "".join([f"<span><strong>{k}</strong>: {v}</span><br>" for k, v in items])

    return f"""
    <div style="margin:8px 0 8px 0;">
      <div style="background:#F5F9FF;border-left:6px solid {_theme.primary_color};
                  padding:14px 16px;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
        <div style="display:flex;gap:14px;align-items:flex-start;">
          <div style="font-size:22px;">{icon}</div>
          <div>
            <div style="font-weight:700;color:{_theme.primary_color};margin-bottom:4px;">{title}</div>
            <div style="color:#000;font-size:14px;line-height:1.5;">{desc}</div>
            {f'<div style="margin-top:8px;color:#000;font-size:13px;">{meta_html}</div>' if meta_html else ''}
          </div>
        </div>
      </div>
    </div>
    """


def render_disclaimer(text: str,
                      icon: str = "⚠️",
                      title: str = "Educational Use Only") -> str:
    """
    Render a disclaimer/warning card.

    Args:
        text: Warning text
        icon: Warning icon/emoji
        title: Warning title

    Returns:
        HTML string for the disclaimer
    """
    return f"""
    <div style="margin:8px 0 6px 0;">
      <div style="background:#FFF4F4;border-left:6px solid {_theme.accent_color};
                  padding:12px 16px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
        <div style="display:flex;gap:10px;align-items:flex-start;color:#000;">
          <span style="font-size:20px">{icon}</span>
          <div>
            <div style="font-weight:700;margin-bottom:4px;">{title}</div>
            <div style="font-size:14px;line-height:1.4;">{text}</div>
          </div>
        </div>
      </div>
    </div>
    """


# Backward compatibility - expose old function names
def set_colors(**kwargs):
    """Legacy function - use configure() instead."""
    _theme.update_colors(**kwargs)


def set_font(font_family: str):
    """Legacy function - use configure() instead."""
    _theme.update_font(font_family)


def set_meta(**kwargs):
    """Legacy function - use configure() instead."""
    _theme.update_meta(**kwargs)


# Expose custom_css as a property for backward compatibility
@property
def custom_css():
    return _theme.get_css()