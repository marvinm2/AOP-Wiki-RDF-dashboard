# AOP-Wiki RDF Dashboard Documentation

This directory contains comprehensive Sphinx documentation for the AOP-Wiki RDF Dashboard project.

## Documentation Features

- **Google/Sphinx format docstrings** throughout all Python modules
- **Professional Sphinx configuration** with VHP4Safety branding
- **Complete API reference** for all functions and classes
- **Quick start guide** and configuration documentation
- **Responsive design** with custom CSS styling

## Building the Documentation

### Prerequisites

Install Sphinx and required extensions:

```bash
pip install sphinx sphinx-rtd-theme
```

### Build HTML Documentation

```bash
# From the docs/ directory
cd docs
make html

# Or from the project root
cd docs && make html
```

The built documentation will be available in `docs/build/html/index.html`.

### Build PDF Documentation (Optional)

```bash
# Requires LaTeX installation
make latexpdf
```

### Clean Build Files

```bash
make clean
```

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main documentation page
│   ├── modules.rst          # API reference
│   ├── quickstart.rst       # Quick start guide
│   ├── configuration.rst    # Configuration guide
│   ├── api.rst              # Detailed API documentation
│   └── _static/
│       └── custom.css       # VHP4Safety styling
├── build/                   # Generated documentation
├── Makefile                 # Build commands
└── README.md                # This file
```

## Viewing the Documentation

After building, open `docs/build/html/index.html` in your web browser to view the complete documentation with:

- Interactive navigation sidebar
- Professional VHP4Safety branding
- Comprehensive API reference with Google-style docstrings
- Code examples and usage guidelines
- Configuration and deployment guides

## Documentation Standards

All Python modules follow Google-style docstrings compatible with Sphinx autodoc:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """Brief description of the function.
    
    Longer description with more details about the function's
    purpose and behavior.
    
    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter with default value
    
    Returns:
        Description of the return value
        
    Raises:
        ValueError: When invalid parameters are provided
        
    Example:
        >>> result = example_function("test", 5)
        >>> print(result)
        True
    """
```

## Contributing to Documentation

When adding new functions or modifying existing ones:

1. **Add comprehensive docstrings** following the Google style format
2. **Include examples** where appropriate
3. **Document all parameters and return values**
4. **Explain error conditions** and exceptions
5. **Rebuild documentation** to verify formatting

## Customization

The documentation uses VHP4Safety branding with the official color palette:

- Primary: #307BBF (blue)
- Secondary: #E6007E (magenta) 
- Accent: #29235C (dark blue)
- Light: #93D5F6 (light blue)
- Content: #FF9500 (orange)

Colors and styling can be customized in `source/_static/custom.css`.