# AOP-Wiki Dashboard User Guide

Welcome to the AOP-Wiki RDF Dashboard! This guide will help you navigate and make the most of the dashboard's features for analyzing Adverse Outcome Pathways (AOPs) data.

## Getting Started

### Accessing the Dashboard
1. Open your web browser
2. Navigate to the dashboard URL (typically `http://localhost:5000`)
3. The dashboard loads automatically with the latest data

### System Requirements
- **Modern Web Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **Internet Connection**: Required for interactive chart features
- **Screen Resolution**: Optimized for desktop (1280px+) but works on mobile

---

## Dashboard Navigation

### Main Tabs
The dashboard features two main sections accessible via the navigation tabs:

#### **Latest Data Tab**
- **Purpose**: View current state of the AOP-Wiki database
- **Content**: Real-time snapshots and quality metrics
- **Best For**: Understanding current data landscape, quality assessment

#### **Historical Trends Tab**  
- **Purpose**: Analyze how the database has evolved over time
- **Content**: Time-series plots showing changes across versions
- **Best For**: Trend analysis, impact assessment, growth tracking

### Quick Navigation Buttons
Within each tab, use the section navigation buttons to jump to specific areas:
- **Database State** → Current entity counts and summaries
- **Network Analysis** → Connectivity and relationship analysis  
- **Data Quality** → Completeness and annotation patterns
- **KE Analysis** → Key Event component breakdowns

---

## Understanding the Visualizations

### **Latest Data Visualizations**

#### **Entity Counts**
- **Shows**: Total AOPs, Key Events (KEs), Key Event Relationships (KERs), Stressors, and Authors
- **Use Case**: Get a quick overview of database size and scope
- **Interpretation**: Higher counts indicate more comprehensive coverage

#### **AOP Connectivity**  
- **Shows**: Connected vs. Isolated AOPs based on shared Key Events
- **Use Case**: Understand network structure and pathway relationships
- **Interpretation**: Higher connectivity suggests more integrated knowledge

#### **KE Component Distribution**
- **Shows**: Breakdown of Process, Object, and Action annotations
- **Use Case**: Assess annotation completeness and patterns
- **Interpretation**: Balanced distribution indicates comprehensive annotation

#### **AOP Completeness**
- **Shows**: Distribution of essential vs. optional property completeness
- **Use Case**: Quality assessment of AOP documentation
- **Interpretation**: Higher completeness indicates better data quality

#### **Biological Usage Patterns**
- **Shows**: Distribution of ontology sources (GO, CHEBI, PR, etc.)
- **Use Case**: Understand which biological domains are most represented
- **Interpretation**: Diversity indicates broad biological coverage

### **Historical Trends Visualizations**

#### **Entity Count Trends**
- **Shows**: Growth of AOPs, KEs, KERs, and Stressors over time
- **Use Case**: Track database growth and development patterns
- **Toggle Options**: Absolute counts vs. delta (changes between versions)

#### **Author Contributions**
- **Shows**: Number of unique authors contributing to AOPs over time
- **Use Case**: Understand community growth and engagement
- **Interpretation**: Increasing authors indicates growing community involvement

#### **AOP Lifecycle Analysis**
- **Shows**: When AOPs were created vs. when they were last modified
- **Use Case**: Understand content development and maintenance patterns
- **Interpretation**: Active modification suggests ongoing curation

---

## Data Export Features

### CSV Download Functionality
Every visualization includes a **"Download CSV"** button that exports the underlying data.

#### **How to Download Data**
1. **Locate the Download Button**: Look for "Download CSV" in the top-right corner of each plot
2. **Click to Download**: Browser will download the CSV file automatically
3. **File Naming**: Files are automatically named (e.g., `latest_entity_counts.csv`)

#### **What's Included in Downloads**
- **Raw Data**: All data points used to create the visualization
- **Version Information**: Database version/date for context
- **Metadata**: Additional context like totals, calculations, and reference values
- **Clean Format**: Ready for analysis in Excel, R, Python, or other tools

#### **Special Download Options**
- **Historical Trends**: Choose between "CSV (Abs)" for absolute values or "CSV (Δ)" for delta changes
- **Rich Context**: All exports include relevant metadata for proper interpretation

### **Use Cases for Data Export**
- **Further Analysis**: Import into statistical software or spreadsheets
- **Reporting**: Create custom reports with specific data subsets
- **Integration**: Use data in other applications or workflows
- **Archival**: Keep snapshots of data for historical reference

---

## Interactive Features

### **Plot Interactions**
- **Hover Details**: Move your mouse over data points to see detailed values
- **Zoom**: Use mouse wheel or plot controls to zoom in/out
- **Pan**: Click and drag to move around zoomed plots
- **Reset**: Double-click to reset zoom level

### **View Toggles**
- **Delta Mode**: Historical trends offer absolute vs. delta (change) views
- **Responsive Layout**: Plots automatically adjust to your screen size
- **Mobile Friendly**: Touch-friendly controls on mobile devices

---

## Mobile Usage

### **Mobile Optimization**
- **Responsive Design**: All visualizations adapt to mobile screens
- **Touch Controls**: Tap and pinch to interact with plots  
- **Simplified Navigation**: Streamlined mobile navigation experience
- **Download Support**: CSV downloads work on mobile browsers

### **Best Practices for Mobile**
- **Portrait Orientation**: Generally provides better plot visibility
- **WiFi Connection**: Recommended for faster loading of interactive elements
- **Modern Browser**: Use latest version of mobile browser for best experience

---

## Troubleshooting

### **Common Issues and Solutions**

#### **Plots Not Loading**
- **Check Internet Connection**: Interactive features require connectivity
- **Refresh Page**: Browser refresh often resolves loading issues
- **Clear Cache**: Browser cache issues can affect plot rendering

#### **Slow Performance**  
- **Close Other Tabs**: Free up browser resources
- **Check Network**: Slow connection affects interactive plot features
- **Wait for Loading**: Initial load includes all data processing

#### **Download Issues**
- **Check Downloads Folder**: Files save to default download location
- **Enable Downloads**: Ensure browser allows automatic downloads
- **Try Right-Click**: Right-click download button → "Save Link As"

#### **Mobile Display Issues**
- **Rotate Screen**: Try both portrait and landscape orientations
- **Zoom Reset**: Double-tap plots to reset zoom level
- **Refresh**: Mobile refresh can resolve display glitches

---

## Best Practices

### **For Data Analysis**
1. **Start with Latest Data**: Get current state overview first
2. **Use Historical Context**: Compare with trends for better insight
3. **Download Key Datasets**: Export data for deeper analysis
4. **Cross-Reference**: Use multiple plots together for comprehensive understanding

### **For Reporting**
1. **Export Visualizations**: Use browser screenshot tools for images
2. **Download Supporting Data**: Include CSV data for transparency
3. **Note Versions**: Always include database version in reports
4. **Provide Context**: Explain what metrics mean for your audience

### **For Monitoring**
1. **Regular Check-ins**: Visit dashboard regularly to track changes
2. **Compare Versions**: Use historical trends to identify significant changes  
3. **Focus on Quality**: Pay attention to completeness and annotation metrics
4. **Track Growth**: Monitor entity counts and author contributions

---

## Getting Help

### **Additional Resources**
- **System Status**: Visit `/status` page for real-time system health
- **Technical Documentation**: See README.md for technical details
- **Issue Reporting**: Contact development team for bugs or feature requests

### **Contact Information**
For technical support or questions about the dashboard:
1. **Check This Guide**: Most common questions are answered here
2. **Review Documentation**: Technical details available in README.md
3. **Contact Team**: Reach out to VHP4Safety development team

---

## Updates and Changes

### **Dashboard Evolution**
The dashboard is continuously improved with:
- **New Visualizations**: Additional analysis types and metrics
- **Enhanced Features**: Improved interactivity and export options  
- **Performance Improvements**: Faster loading and better reliability
- **User Interface Updates**: Better navigation and user experience

### **Staying Updated**
- **Regular Visits**: Dashboard reflects latest AOP-Wiki data automatically
- **Feature Announcements**: Major updates communicated through appropriate channels
- **Feedback Welcome**: User suggestions drive dashboard improvements

---

**Happy analyzing! The AOP-Wiki Dashboard is your window into the evolving world of Adverse Outcome Pathways research.**