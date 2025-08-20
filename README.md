# AOP-Wiki RDF Dashboard

A comprehensive web dashboard for monitoring and analyzing the AOP-Wiki RDF knowledge base evolution over time. Built for the VHP4Safety project to provide insights into Adverse Outcome Pathways (AOPs) data trends, quality metrics, and structural analysis.

## Features

### **Interactive Visualizations**
- **Latest Data Overview**: Current state snapshots of the AOP-Wiki database
- **Historical Trends**: Time-series analysis of database evolution
- **Network Analysis**: AOP connectivity and structural relationships
- **Data Quality Metrics**: Annotation completeness and ontology usage patterns

### **Comprehensive Analytics**
- Entity counts (AOPs, Key Events, Key Event Relationships, Stressors)
- Key Event component distribution and annotation depth
- Biological process and object ontology usage patterns
- AOP completeness analysis using configurable property types
- Network density and connectivity analysis
- Author contribution tracking over time

### **Data Export Capabilities**
- **CSV Download**: Export underlying data from any plot for further analysis
- **10+ Download Options**: Entity counts, components, usage patterns, trends, and more
- **Multiple Formats**: Absolute values and delta (change) datasets available
- **Rich Metadata**: All exports include version info and contextual data

### **Performance & Reliability**
- **Parallel Processing**: Concurrent plot generation for fast loading
- **Health Monitoring**: Real-time SPARQL endpoint status tracking
- **Error Handling**: Graceful degradation with fallback visualizations
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## Architecture

### **Backend Components**
- **Flask Web Framework**: Lightweight and fast web server
- **SPARQL Integration**: Direct queries to AOP-Wiki RDF endpoint
- **Plotly Visualizations**: Interactive, publication-quality charts
- **Pandas Data Processing**: Efficient data manipulation and caching
- **Concurrent Execution**: Multi-threaded plot generation

### **Data Sources**
- **Primary**: AOP-Wiki SPARQL endpoint (`http://localhost:8890/sparql`)
- **Configuration**: Property metadata from `property_labels.csv`
- **Ontology Sources**: GO, CHEBI, PR, CL, UBERON, HP, MP, and others

### **Visualization Types**
- Time-series line plots for trends
- Bar charts for entity counts and metrics
- Pie charts for distributions and proportions
- Interactive hover details and responsive layouts

## Quick Start

### Prerequisites
```bash
# Python 3.8+ required
python3 --version

# Install dependencies
pip install -r requirements.txt
```

### Configuration
1. **SPARQL Endpoint**: Ensure AOP-Wiki RDF endpoint is accessible
2. **Property Labels**: Update `property_labels.csv` with desired AOP properties
3. **Environment**: Configure settings in `config.py`

### Running the Dashboard
```bash
# Start the application
python app.py

# Access the dashboard
open http://localhost:5000
```

### Health Check
```bash
# Verify system status
curl http://localhost:5000/health

# Monitor real-time status
open http://localhost:5000/status
```

## Usage Guide

### **Navigation**
- **Latest Data**: Current database snapshots and quality metrics
- **Historical Trends**: Time-series analysis and evolution patterns

### **Interactive Features**
- **Toggle Views**: Switch between absolute values and deltas
- **CSV Downloads**: Export data from any plot using download buttons
- **Responsive Charts**: Hover for details, zoom, and pan capabilities

### **Data Export**
1. Navigate to any plot with a "Download CSV" button
2. Click to download the underlying dataset
3. For historical trends: choose "CSV (Abs)" or "CSV (Δ)"
4. Files include version metadata and contextual information

## Development

### **File Structure**
```
├── app.py                 # Main Flask application
├── plots.py               # Plot generation and data processing
├── config.py              # Configuration settings
├── property_labels.csv    # AOP property metadata
├── templates/             # HTML templates
│   ├── index.html        # Main dashboard layout
│   ├── trends.html       # Historical trends section
│   └── status.html       # System monitoring page
├── static/
│   └── css/main.css      # Styling and responsive design
├── docs/                  # Comprehensive Sphinx documentation
│   ├── source/           # Documentation source files
│   │   ├── conf.py      # Sphinx configuration
│   │   ├── index.rst    # Main documentation page
│   │   ├── modules.rst  # API reference
│   │   ├── quickstart.rst # Quick start guide
│   │   ├── configuration.rst # Configuration guide
│   │   └── api.rst      # Detailed API documentation
│   └── build/            # Generated documentation
└── requirements.txt      # Python dependencies
```

### **Adding New Plots**
1. Create plot function in `plots.py` with data caching
2. Add Flask route for CSV download in `app.py`
3. Include plot variable in template rendering
4. Add download button to appropriate HTML template

### **Customization**
- **Colors**: Modify `BRAND_COLORS` in `plots.py`
- **Properties**: Update `property_labels.csv` with new AOP properties
- **Endpoints**: Configure SPARQL endpoint in `config.py`
- **Layout**: Customize templates and CSS for different layouts

## Configuration

### **Environment Variables**
```bash
SPARQL_ENDPOINT=http://localhost:8890/sparql
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
LOG_LEVEL=INFO
PARALLEL_WORKERS=5
PLOT_TIMEOUT=60
```

### **Property Configuration**
Edit `property_labels.csv` to customize AOP completeness analysis:
```csv
uri,label,type
http://aopkb.org/aop_ontology#has_key_event,Has Key Event,Content
http://purl.org/dc/elements/1.1/title,Title,Essential
```

## Troubleshooting

### **Common Issues**
- **Plots not loading**: Check SPARQL endpoint connectivity
- **Slow performance**: Adjust `PARALLEL_WORKERS` and `PLOT_TIMEOUT`
- **CSV downloads failing**: Verify data caching is working correctly
- **Memory issues**: Reduce concurrent workers or increase system resources

### **Logging**
Monitor application logs for detailed error information:
```bash
tail -f app_new.log
```

### **Health Monitoring**
Use the built-in monitoring endpoints:
- `/health` - System health check
- `/status` - Real-time monitoring dashboard

## Contributing

### **Development Setup**
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation
5. Submit a pull request

### **Code Standards**
- Follow PEP 8 for Python code
- Include comprehensive Google/Sphinx style docstrings
- Use type hints for all functions and methods
- Test all SPARQL queries and visualizations
- Ensure responsive design for new UI elements

### **Documentation**
The project includes comprehensive Google/Sphinx format documentation:

#### **Building Documentation**
```bash
# Install Sphinx dependencies
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs && make html

# View documentation
open docs/build/html/index.html
```

#### **Documentation Features**
- **Professional Sphinx Documentation**: Complete API reference with VHP4Safety branding
- **Google-Style Docstrings**: All functions documented with comprehensive examples
- **Interactive API Reference**: Auto-generated from docstrings with cross-references
- **Configuration Guides**: Detailed environment variable and deployment documentation
- **Quick Start Guides**: Step-by-step setup and usage instructions

## License

Todo

## Support

For questions, issues, or feature requests:
1. Check the troubleshooting section above
2. Review existing issues in the project repository
3. Contact the VHP4Safety development team

## Related Projects

- **VHP4Safety**: Virtual Human Platform for safety assessment
- **AOP-Wiki**: Adverse Outcome Pathways knowledge base

