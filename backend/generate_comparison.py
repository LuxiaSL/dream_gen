"""
Enhanced HTML Comparison Generator with Clean Template Architecture

This module provides a cleaner approach to HTML generation using atomic building
blocks instead of giant f-strings. This makes it easier to manage, less prone
to errors, and more maintainable.
"""

import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


class HTMLTemplateBuilder:
    """
    Builds HTML templates using an atomic approach with separate components
    for CSS, JavaScript, and HTML content.
    """
    
    def __init__(self):
        self.css_blocks = []
        self.js_blocks = []
        self.html_blocks = []
        self.js_data = {}
        
    def add_css(self, css: str):
        """Add a CSS block to the template"""
        self.css_blocks.append(css.strip())
        
    def add_js(self, js: str):
        """Add a JavaScript block to the template"""
        self.js_blocks.append(js.strip())
        
    def add_html(self, html: str):
        """Add an HTML block to the template"""
        self.html_blocks.append(html.strip())
        
    def add_js_data(self, key: str, value: Any):
        """Add data that will be available to JavaScript"""
        self.js_data[key] = value
        
    def build_css_section(self) -> str:
        """Build the complete CSS section"""
        return "\n\n".join(self.css_blocks)
        
    def build_js_data_section(self) -> str:
        """Build JavaScript data declarations"""
        js_lines = []
        for key, value in self.js_data.items():
            if isinstance(value, (list, dict)):
                js_lines.append(f"const {key} = {json.dumps(value)};")
            elif isinstance(value, str):
                js_lines.append(f"const {key} = {json.dumps(value)};")
            else:
                js_lines.append(f"const {key} = {value};")
        return "\n        ".join(js_lines)
        
    def build_js_section(self) -> str:
        """Build the complete JavaScript section"""
        data_section = self.build_js_data_section()
        code_section = "\n\n".join(self.js_blocks)
        return f"{data_section}\n\n{code_section}"
        
    def build_html_section(self) -> str:
        """Build the complete HTML body content"""
        return "\n".join(self.html_blocks)
        
    def build(self, title: str = "Document") -> str:
        """Build the complete HTML document"""
        template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
{self.build_css_section()}
    </style>
</head>
<body>
{self.build_html_section()}
    
    <script>
        {self.build_js_section()}
    </script>
</body>
</html>"""
        return template


def generate_base_css() -> str:
    """Generate base CSS styles"""
    return """
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding-bottom: 50px;
        }
        
        .container {
            max-width: 1920px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        }
        
        h1 {
            font-size: 3em;
            margin-bottom: 15px;
            color: #fff;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
        }
        
        .test-info {
            font-size: 1.1em;
            color: #ddd;
            margin-top: 15px;
        }"""


def generate_component_css() -> str:
    """Generate component-specific CSS styles"""
    return """
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .tab {
            padding: 15px 30px;
            background: #2d3748;
            border: none;
            border-radius: 10px;
            color: #e0e0e0;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }
        
        .tab:hover {
            background: #4a5568;
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.4);
        }
        
        .tab.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
        }
        
        .tab-content {
            display: none;
            animation: fadeIn 0.3s;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .section {
            background: #2d3748;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }"""


def generate_chart_css() -> str:
    """Generate chart-specific CSS"""
    return """
        .chart-container {
            position: relative;
            height: 500px;
            margin: 20px 0;
        }
        
        .controls {
            background: #2d3748;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .control-group {
            margin-bottom: 20px;
        }
        
        .control-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #667eea;
            font-size: 1.1em;
        }
        
        .control-group input[type="range"] {
            width: 100%;
            height: 8px;
            background: #4a5568;
            border-radius: 5px;
            outline: none;
            -webkit-appearance: none;
        }
        
        .control-group input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #667eea;
            cursor: pointer;
            border-radius: 50%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }"""


def generate_card_css() -> str:
    """Generate card and comparison grid CSS"""
    return """
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 25px;
        }
        
        .config-card {
            background: #2d3748;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            transition: all 0.3s;
            border: 2px solid transparent;
        }
        
        .config-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
            border-color: #667eea;
        }
        
        .config-card h3 {
            font-size: 1.2em;
            margin-bottom: 12px;
            color: #667eea;
            word-wrap: break-word;
        }
        
        .config-card img {
            width: 100%;
            height: auto;
            border-radius: 10px;
            margin: 12px 0;
            border: 3px solid #4a5568;
            transition: all 0.3s;
            cursor: zoom-in;
        }
        
        .config-card img:hover {
            border-color: #667eea;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
        }"""


def generate_utility_css() -> str:
    """Generate utility and misc CSS"""
    return """
        .metrics {
            font-size: 0.95em;
            line-height: 1.8;
        }
        
        .metrics div {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #4a5568;
        }
        
        .metrics .label {
            color: #a0aec0;
        }
        
        .metrics .value {
            font-weight: bold;
            color: #e0e0e0;
        }
        
        .badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 5px;
            font-size: 0.85em;
            font-weight: bold;
            margin-top: 8px;
        }
        
        .badge-baseline { background: #3498db; color: white; }
        .badge-target { background: #2ecc71; color: white; }
        .badge-close { background: #f39c12; color: white; }
        .badge-recommended { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 8px 15px; }
        .badge-multi { background: #e74c3c; color: white; }
        
        .fps-excellent { color: #2ecc71; font-weight: bold; }
        .fps-good { color: #3498db; }
        .fps-medium { color: #f39c12; }
        .fps-low { color: #e74c3c; }
        
        .quality-excellent { color: #2ecc71; }
        .quality-good { color: #3498db; }
        .quality-medium { color: #f39c12; }
        .quality-poor { color: #e74c3c; }
        
        .frame-info {
            text-align: center;
            margin-top: 12px;
            font-size: 1.2em;
            color: #a0aec0;
            font-weight: bold;
        }
        
        .frame-info span {
            color: #667eea;
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .summary-card h3 {
            color: #ddd;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .summary-card .value {
            font-size: 2.5em;
            color: white;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }"""


def generate_table_css() -> str:
    """Generate table CSS"""
    return """
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
            background: #1a202c;
            border-radius: 10px;
            overflow: hidden;
        }
        
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #4a5568;
        }
        
        th {
            background: #1a202c;
            color: #667eea;
            font-weight: bold;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        tr:hover {
            background: #374151;
        }
        
        .highlight-row {
            background: rgba(102, 126, 234, 0.2);
            border-left: 4px solid #667eea;
        }"""


def generate_modal_css() -> str:
    """Generate modal CSS"""
    return """
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.95);
            justify-content: center;
            align-items: center;
        }
        
        .modal-content {
            max-width: 95%;
            max-height: 95%;
            position: relative;
        }
        
        .modal-content img {
            width: 100%;
            height: auto;
            border-radius: 10px;
        }
        
        .close-modal {
            position: absolute;
            top: 20px;
            right: 40px;
            color: #fff;
            font-size: 50px;
            font-weight: bold;
            cursor: pointer;
            text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
            z-index: 1001;
        }
        
        .close-modal:hover {
            color: #667eea;
        }"""


def generate_misc_css() -> str:
    """Generate miscellaneous CSS"""
    return """
        .filter-controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .filter-controls select {
            padding: 10px 20px;
            background: #1a202c;
            border: 2px solid #4a5568;
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 1em;
            cursor: pointer;
        }
        
        footer {
            text-align: center;
            padding: 30px;
            color: #a0aec0;
            font-size: 0.95em;
            border-top: 1px solid #4a5568;
            margin-top: 50px;
        }
        
        @media (max-width: 768px) {
            .comparison-grid {
                grid-template-columns: 1fr;
            }
            
            h1 {
                font-size: 2em;
            }
            
            .chart-container {
                height: 300px;
            }
        }"""


def generate_scatter_chart_js(unique_divisors: list, color_map: dict) -> str:
    """Generate scatter chart JavaScript code"""
    js_code = """
        // Scatter Plot
        const scatterData = configNames.map((name, i) => ({
            x: fpsValues[i],
            y: ssimValues[i],
            label: name,
            divisor: divisors[i]
        }));
        
        new Chart(document.getElementById('scatterChart'), {
            type: 'scatter',
            data: {
                datasets: ["""
    
    # Add datasets for each divisor
    dataset_entries = []
    for divisor in unique_divisors:
        color = color_map[divisor]
        dataset_entries.append(f"""
                    {{
                        label: 'Divisor {divisor}',
                        data: scatterData.filter(d => d.divisor === {divisor}),
                        backgroundColor: '{color}80',
                        borderColor: '{color}',
                        borderWidth: 2
                    }}""")
    
    js_code += ",".join(dataset_entries) + """
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'FPS vs SSIM Quality (Higher is Better on Both Axes)',
                        font: { size: 16 }
                    },
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                xMin: targetFPS,
                                xMax: targetFPS,
                                borderColor: '#f39c12',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    display: true,
                                    content: 'Target: 15 FPS',
                                    position: 'start'
                                }
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'FPS (Frames Per Second)', font: { size: 14 } },
                        grid: { color: '#4a5568' }
                    },
                    y: {
                        title: { display: true, text: 'SSIM Quality (0-1)', font: { size: 14 } },
                        grid: { color: '#4a5568' }
                    }
                }
            }
        });"""
    
    return js_code


def generate_fps_chart_js() -> str:
    """Generate FPS bar chart JavaScript code"""
    return """
        // FPS Bar Chart
        new Chart(document.getElementById('fpsChart'), {
            type: 'bar',
            data: {
                labels: configNames,
                datasets: [{
                    label: 'FPS',
                    data: fpsValues,
                    backgroundColor: colors.map(c => c + '80'),
                    borderColor: colors,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'FPS Performance by Configuration',
                        font: { size: 16 }
                    },
                    legend: { display: false }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'FPS', font: { size: 14 } },
                        beginAtZero: true,
                        grid: { color: '#4a5568' }
                    },
                    x: {
                        display: false
                    }
                }
            }
        });"""


def generate_divisor_chart_js() -> str:
    """Generate divisor chart JavaScript code"""
    return """
        // FPS by Divisor (Box Plot style - using average per divisor)
        const avgFPSByDivisor = uniqueDivisors.map(div => {
            const values = fpsValues.filter((_, i) => divisors[i] === div);
            return values.reduce((a, b) => a + b, 0) / values.length;
        });
        
        // Generate colors for divisor chart
        const divisorColors = uniqueDivisors.map(div => {
            return colorMap[div] || '#95a5a6';
        });
        
        new Chart(document.getElementById('divisorChart'), {
            type: 'bar',
            data: {
                labels: uniqueDivisors.map(d => `Divisor ${d}`),
                datasets: [{
                    label: 'Average FPS',
                    data: avgFPSByDivisor,
                    backgroundColor: divisorColors.map(c => c + '80'),
                    borderColor: divisorColors,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Average FPS by Resolution Divisor',
                        font: { size: 16 }
                    }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Average FPS', font: { size: 14 } },
                        beginAtZero: true,
                        grid: { color: '#4a5568' }
                    }
                }
            }
        });"""


def generate_quality_charts_js() -> str:
    """Generate SSIM and PSNR charts JavaScript code"""
    return """
        // SSIM Chart
        new Chart(document.getElementById('ssimChart'), {
            type: 'bar',
            data: {
                labels: configNames,
                datasets: [{
                    label: 'SSIM',
                    data: ssimValues,
                    backgroundColor: colors.map(c => c + '80'),
                    borderColor: colors,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'SSIM Quality Scores (Structural Similarity)',
                        font: { size: 16 }
                    },
                    legend: { display: false }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'SSIM (0-1, higher is better)', font: { size: 14 } },
                        beginAtZero: true,
                        max: 1,
                        grid: { color: '#4a5568' }
                    },
                    x: {
                        display: false
                    }
                }
            }
        });
        
        // PSNR Chart
        new Chart(document.getElementById('psnrChart'), {
            type: 'bar',
            data: {
                labels: configNames,
                datasets: [{
                    label: 'PSNR',
                    data: psnrValues,
                    backgroundColor: colors.map(c => c + '80'),
                    borderColor: colors,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'PSNR Quality Scores (Peak Signal-to-Noise Ratio)',
                        font: { size: 16 }
                    },
                    legend: { display: false }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'PSNR (dB, higher is better)', font: { size: 14 } },
                        beginAtZero: true,
                        grid: { color: '#4a5568' }
                    },
                    x: {
                        display: false
                    }
                }
            }
        });"""


def generate_utility_js() -> str:
    """Generate utility JavaScript functions"""
    return """
        // Tab switching
        function showTab(tabName) {
            const tabs = document.querySelectorAll('.tab');
            const contents = document.querySelectorAll('.tab-content');
            
            tabs.forEach(tab => tab.classList.remove('active'));
            contents.forEach(content => content.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        // Frame selector
        const frameSelector = document.getElementById('frame-selector');
        const frameNumber = document.getElementById('frame-number');
        const configImages = document.querySelectorAll('.config-image');
        
        frameSelector.addEventListener('input', function() {
            const frame = parseInt(this.value);
            frameNumber.textContent = frame;
            
            configImages.forEach(img => {
                const configName = img.dataset.config;
                const framePath = `${configName}/frame_${String(frame).padStart(3, '0')}.png`;
                img.src = framePath;
            });
        });
        
        // Filter by divisor
        function filterConfigs() {
            const filter = document.getElementById('divisor-filter').value;
            const cards = document.querySelectorAll('.config-card');
            
            cards.forEach(card => {
                if (filter === 'all' || String(card.dataset.divisor) === String(filter)) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }
        
        // Modal functionality
        function openModal(src, title) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modal-image');
            const modalTitle = document.getElementById('modal-title');
            modal.style.display = 'flex';
            modalImg.src = src;
            modalTitle.textContent = title;
        }
        
        function closeModal(event) {
            if (event) event.stopPropagation();
            document.getElementById('imageModal').style.display = 'none';
        }
        
        // Keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeModal();
            } else if (e.key === 'ArrowLeft') {
                frameSelector.value = Math.max(0, parseInt(frameSelector.value) - 1);
                frameSelector.dispatchEvent(new Event('input'));
            } else if (e.key === 'ArrowRight') {
                frameSelector.value = Math.min(maxFrameIndex, parseInt(frameSelector.value) + 1);
                frameSelector.dispatchEvent(new Event('input'));
            }
        });
        
        // Initialize with divisor 4 filter
        filterConfigs();"""


def build_color_map(unique_divisors: list) -> dict:
    """Build a dynamic color map for divisors"""
    base_colors = [
        '#3498db',  # Blue
        '#9b59b6',  # Purple
        '#1abc9c',  # Turquoise
        '#2ecc71',  # Green
        '#e74c3c',  # Red
        '#f39c12',  # Orange
        '#34495e',  # Dark gray-blue
        '#16a085',  # Teal
        '#c0392b',  # Dark red
        '#8e44ad'   # Dark purple
    ]
    
    color_map = {}
    for i, divisor in enumerate(unique_divisors):
        # Special case: div1 is always blue (baseline)
        if divisor == 1:
            color_map[divisor] = '#3498db'
        # Special case: div4 is typically recommended, use green
        elif divisor == 4:
            color_map[divisor] = '#2ecc71'
        else:
            # Assign colors cyclically
            color_idx = i % len(base_colors)
            color_map[divisor] = base_colors[color_idx]
    
    return color_map


def generate_comparison_html(test_dir: Path, output_path: Path = None):
    """
    Generate enhanced interactive HTML comparison page using clean template architecture
    """
    if output_path is None:
        output_path = test_dir / "comparison_enhanced.html"
    
    # Load performance metrics
    metrics_file = test_dir / "performance_metrics.json"
    if not metrics_file.exists():
        print(f"[ERROR] Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    test_info = data['test_info']
    results = data['results']
    
    # Get the actual number of frames generated
    num_frames = results[0]['performance']['frames_generated'] if results else 20
    max_frame_index = num_frames - 1
    
    # Get unique divisors and build color map
    unique_divisors = sorted(list(set(r['config']['divisor'] for r in results)))
    color_map = build_color_map(unique_divisors)
    
    # Prepare data for charts
    config_names = []
    fps_values = []
    ssim_values = []
    psnr_values = []
    divisors = []
    colors = []
    
    for result in results:
        config = result['config']
        perf = result['performance']
        quality = result['quality']
        
        config_names.append(config['name'])
        fps_values.append(round(perf['fps'], 2))
        divisors.append(config['divisor'])
        colors.append(color_map.get(config['divisor'], '#95a5a6'))
        
        ssim = quality['avg_ssim'] if quality['avg_ssim'] else 0
        psnr = quality['avg_psnr'] if quality['avg_psnr'] else 0
        ssim_values.append(round(ssim, 4))
        psnr_values.append(round(psnr, 2))
    
    # Calculate recommended config dynamically
    target_fps = test_info.get('target_fps', 15.0)
    recommended_configs = [r for r in results if r['performance']['fps'] >= target_fps]
    
    if recommended_configs:
        recommended_configs.sort(key=lambda x: x['quality']['avg_ssim'] or 0, reverse=True)
        recommended = recommended_configs[0]
        recommended_text = f"{recommended['config']['name']} ({recommended['performance']['fps']:.1f} FPS, SSIM {recommended['quality']['avg_ssim']:.2f})"
    else:
        sorted_by_fps = sorted(results, key=lambda x: x['performance']['fps'], reverse=True)
        top_quarter = sorted_by_fps[:max(1, len(sorted_by_fps) // 4)]
        best_balanced = max(top_quarter, key=lambda x: x['quality']['avg_ssim'] or 0)
        recommended_text = f"{best_balanced['config']['name']} ({best_balanced['performance']['fps']:.1f} FPS, SSIM {best_balanced['quality']['avg_ssim']:.2f})"
    
    # Create template builder
    builder = HTMLTemplateBuilder()
    
    # Add CSS blocks
    builder.add_css(generate_base_css())
    builder.add_css(generate_component_css())
    builder.add_css(generate_chart_css())
    builder.add_css(generate_card_css())
    builder.add_css(generate_utility_css())
    builder.add_css(generate_table_css())
    builder.add_css(generate_modal_css())
    builder.add_css(generate_misc_css())
    
    # Add JavaScript data
    builder.add_js_data('configNames', config_names)
    builder.add_js_data('fpsValues', fps_values)
    builder.add_js_data('ssimValues', ssim_values)
    builder.add_js_data('psnrValues', psnr_values)
    builder.add_js_data('divisors', divisors)
    builder.add_js_data('colors', colors)
    builder.add_js_data('targetFPS', target_fps)
    builder.add_js_data('uniqueDivisors', unique_divisors)
    builder.add_js_data('colorMap', color_map)
    builder.add_js_data('maxFrameIndex', max_frame_index)
    
    # Add Chart.js default settings
    builder.add_js("Chart.defaults.color = '#e0e0e0';")
    builder.add_js("Chart.defaults.borderColor = '#4a5568';")
    
    # Add chart JavaScript
    builder.add_js(generate_scatter_chart_js(unique_divisors, color_map))
    builder.add_js(generate_fps_chart_js())
    builder.add_js(generate_divisor_chart_js())
    builder.add_js(generate_quality_charts_js())
    builder.add_js(generate_utility_js())
    
    # Build HTML content
    header_html = f"""
    <div class="container">
        <header>
            <h1>🎨 Interpolation Quality Analysis</h1>
            <div class="test-info">
                <div><strong>Test Date:</strong> {test_info['timestamp']}</div>
                <div><strong>Configurations Tested:</strong> {test_info['total_configs']}</div>
                <div><strong>Target FPS:</strong> {test_info['target_fps']}</div>
                <div><strong>Frames per Config:</strong> {num_frames}</div>
            </div>
        </header>"""
    builder.add_html(header_html)
    
    # Summary cards
    summary_html = f"""
        <div class="summary-cards">
            <div class="summary-card">
                <h3>Total Configs</h3>
                <div class="value">{test_info['total_configs']}</div>
            </div>
            <div class="summary-card">
                <h3>Meeting Target</h3>
                <div class="value">{sum(1 for r in results if r['performance']['fps'] >= test_info['target_fps'])}</div>
            </div>
            <div class="summary-card">
                <h3>Best FPS</h3>
                <div class="value">{max(r['performance']['fps'] for r in results):.1f}</div>
            </div>
            <div class="summary-card">
                <h3>Best Quality</h3>
                <div class="value">{max((r['quality']['avg_ssim'] or 0) for r in results):.3f}</div>
            </div>
        </div>"""
    builder.add_html(summary_html)
    
    # Tabs
    tabs_html = """
        <div class="tabs">
            <button class="tab active" onclick="showTab('overview')">📊 Overview</button>
            <button class="tab" onclick="showTab('performance')">⚡ Performance Charts</button>
            <button class="tab" onclick="showTab('quality')">🎯 Quality Analysis</button>
            <button class="tab" onclick="showTab('comparison')">🖼️ Visual Comparison</button>
            <button class="tab" onclick="showTab('data')">📋 Data Table</button>
        </div>"""
    builder.add_html(tabs_html)
    
    # Overview Tab
    overview_html = f"""
        <div id="overview" class="tab-content active">
            <div class="section">
                <h2>Performance vs Quality Scatter Plot</h2>
                <div class="chart-container">
                    <canvas id="scatterChart"></canvas>
                </div>
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                <div style="font-size: 1.1em; line-height: 1.8;">
                    <p style="margin-bottom: 15px;">✅ <strong>{sum(1 for r in results if r['performance']['fps'] >= test_info['target_fps'])}</strong> configurations meet the {target_fps} FPS target</p>
                    <p style="margin-bottom: 15px;">🏆 <strong>Best Performance:</strong> {max(results, key=lambda r: r['performance']['fps'])['config']['name']} at {max(r['performance']['fps'] for r in results):.2f} FPS</p>
                    <p style="margin-bottom: 15px;">🎨 <strong>Best Quality:</strong> {max(results, key=lambda r: r['quality']['avg_ssim'] or 0)['config']['name']} with SSIM {max((r['quality']['avg_ssim'] or 0) for r in results):.4f}</p>
                    <p style="margin-bottom: 15px;">⚖️ <strong>Recommended:</strong> {recommended_text}</p>
                </div>
            </div>
        </div>"""
    builder.add_html(overview_html)
    
    # Performance Tab
    performance_html = """
        <div id="performance" class="tab-content">
            <div class="section">
                <h2>FPS Comparison by Configuration</h2>
                <div class="chart-container">
                    <canvas id="fpsChart"></canvas>
                </div>
            </div>
            
            <div class="section">
                <h2>FPS by Divisor</h2>
                <div class="chart-container">
                    <canvas id="divisorChart"></canvas>
                </div>
            </div>
        </div>"""
    builder.add_html(performance_html)
    
    # Quality Tab
    quality_html = """
        <div id="quality" class="tab-content">
            <div class="section">
                <h2>SSIM Quality Scores</h2>
                <div class="chart-container">
                    <canvas id="ssimChart"></canvas>
                </div>
            </div>
            
            <div class="section">
                <h2>PSNR Quality Scores</h2>
                <div class="chart-container">
                    <canvas id="psnrChart"></canvas>
                </div>
            </div>
        </div>"""
    builder.add_html(quality_html)
    
    # Visual Comparison Tab
    comparison_html = f"""
        <div id="comparison" class="tab-content">
            <div class="controls">
                <h2>Frame Selector</h2>
                <div class="control-group">
                    <label for="frame-selector">Select Frame to Compare (0-{max_frame_index}):</label>
                    <input type="range" id="frame-selector" min="0" max="{max_frame_index}" value="{min(10, max_frame_index)}" step="1">
                    <div class="frame-info">Frame: <span id="frame-number">{min(10, max_frame_index)}</span> / {max_frame_index}</div>
                </div>
                
                <div class="filter-controls">
                    <select id="divisor-filter" onchange="filterConfigs()">
                        <option value="all">All Divisors</option>"""
    
    # Add divisor filter options
    for divisor in unique_divisors:
        selected = 'selected' if divisor == 4 else ''
        label = f"Divisor {divisor}"
        if divisor == 1:
            label += " (baseline)"
        elif divisor == 4:
            label += " (recommended)"
        comparison_html += f'\n                        <option value="{divisor}" {selected}>{label}</option>'
    
    comparison_html += """
                    </select>
                </div>
            </div>
            
            <div class="comparison-grid" id="comparison-grid">"""
    
    builder.add_html(comparison_html)
    
    # Add config cards
    for i, result in enumerate(results):
        config = result['config']
        perf = result['performance']
        quality = result['quality']
        
        # Determine badges
        badges = []
        if config['divisor'] == 1:
            badges.append('<span class="badge badge-baseline">BASELINE</span>')
        if config.get('multi_upscale'):
            badges.append(f'<span class="badge badge-multi">MULTI-UPSCALE {config["multi_upscale"]}x</span>')
        if recommended_configs and config['name'] == recommended_configs[0]['config']['name']:
            badges.append('<span class="badge badge-recommended">⭐ RECOMMENDED</span>')
        elif perf['fps'] >= target_fps:
            badges.append('<span class="badge badge-target">HITS TARGET</span>')
        elif perf['fps'] >= target_fps - 3:
            badges.append('<span class="badge badge-close">CLOSE</span>')
        
        # Frame path
        frame_rel_path = f"{config['name']}/frame_010.png"
        
        # FPS and quality classes
        fps_class = 'fps-excellent' if perf['fps'] >= 20 else ('fps-good' if perf['fps'] >= 15 else ('fps-medium' if perf['fps'] >= 10 else 'fps-low'))
        
        ssim = quality['avg_ssim'] if quality['avg_ssim'] else 0
        quality_class = 'quality-excellent' if ssim >= 0.8 else ('quality-good' if ssim >= 0.7 else ('quality-medium' if ssim >= 0.6 else 'quality-poor'))
        
        card_html = f"""
                <div class="config-card" data-divisor="{config['divisor']}">
                    <h3>{config['name']}</h3>
                    {''.join(badges)}
                    <img src="{frame_rel_path}" alt="{config['name']}" class="config-image" data-config="{config['name']}" onclick="openModal(this.src, '{config['name']}')">
                    <div class="metrics">
                        <div><span class="label">FPS:</span> <span class="value {fps_class}">{perf['fps']:.2f}</span></div>
                        <div><span class="label">Time:</span> <span class="value">{perf['avg_total_time_ms']:.1f}ms</span></div>
                        <div><span class="label">Divisor:</span> <span class="value">1/{config['divisor']}</span></div>"""
        
        if quality['avg_psnr']:
            card_html += f"""
                        <div><span class="label">PSNR:</span> <span class="value {quality_class}">{quality['avg_psnr']:.1f} dB</span></div>
                        <div><span class="label">SSIM:</span> <span class="value {quality_class}">{quality['avg_ssim']:.4f}</span></div>"""
        
        card_html += """
                    </div>
                </div>"""
        
        builder.add_html(card_html)
    
    builder.add_html("""
            </div>
        </div>""")
    
    # Data Table Tab
    table_html = """
        <div id="data" class="tab-content">
            <div class="section">
                <h2>Complete Performance Data</h2>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Configuration</th>
                                <th>Divisor</th>
                                <th>Down Method</th>
                                <th>Up Method</th>
                                <th>FPS</th>
                                <th>Time (ms)</th>
                                <th>PSNR (dB)</th>
                                <th>SSIM</th>
                            </tr>
                        </thead>
                        <tbody>"""
    
    builder.add_html(table_html)
    
    # Add table rows
    for result in results:
        config = result['config']
        perf = result['performance']
        quality = result['quality']
        
        fps_class = 'fps-excellent' if perf['fps'] >= 20 else ('fps-good' if perf['fps'] >= 15 else ('fps-medium' if perf['fps'] >= 10 else 'fps-low'))
        
        ssim = quality['avg_ssim'] if quality['avg_ssim'] else 0
        quality_class = 'quality-excellent' if ssim >= 0.8 else ('quality-good' if ssim >= 0.7 else ('quality-medium' if ssim >= 0.6 else 'quality-poor'))
        
        psnr_str = f"{quality['avg_psnr']:.1f}" if quality['avg_psnr'] else "N/A"
        ssim_str = f"{quality['avg_ssim']:.4f}" if quality['avg_ssim'] else "N/A"
        
        row_class = ''
        if recommended_configs and config['name'] == recommended_configs[0]['config']['name']:
            row_class = 'highlight-row'
        
        row_html = f"""
                            <tr class="{row_class}">
                                <td><strong>{config['name']}</strong></td>
                                <td>{config['divisor']}</td>
                                <td>{config['downsample'] or 'N/A'}</td>
                                <td>{config['upscale'] or 'N/A'}</td>
                                <td class="{fps_class}">{perf['fps']:.2f}</td>
                                <td>{perf['avg_total_time_ms']:.1f}</td>
                                <td class="{quality_class}">{psnr_str}</td>
                                <td class="{quality_class}">{ssim_str}</td>
                            </tr>"""
        
        builder.add_html(row_html)
    
    builder.add_html("""
                        </tbody>
                    </table>
                </div>
            </div>
        </div>""")
    
    # Modal
    modal_html = """
        <div id="imageModal" class="modal" onclick="closeModal()">
            <span class="close-modal" onclick="closeModal(event)">&times;</span>
            <div class="modal-content">
                <img id="modal-image" src="">
                <div style="text-align: center; color: white; font-size: 1.5em; margin-top: 20px;" id="modal-title"></div>
            </div>
        </div>"""
    builder.add_html(modal_html)
    
    # Footer
    footer_html = f"""
        <footer>
            <p><strong>Dream Window Quality Comparison Tool</strong></p>
            <p>Use keyboard arrows (← →) to navigate frames | Click images for full-screen view</p>
            <p>Test completed: {test_info['timestamp']}</p>
        </footer>
    </div>"""
    builder.add_html(footer_html)
    
    # Build final HTML
    html = builder.build(title="Interpolation Quality Analysis")
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"[OK] Enhanced comparison HTML generated: {output_path}")
    print(f"     Open in browser to view interactive charts and comparisons")
    
    return output_path


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description='Generate enhanced HTML comparison for quality tests')
    parser.add_argument('test_dir', type=Path, help='Test results directory')
    parser.add_argument('--output', type=Path, help='Output HTML file path')
    
    args = parser.parse_args()
    
    if not args.test_dir.exists():
        print(f"[ERROR] Test directory not found: {args.test_dir}")
        return 1
    
    generate_comparison_html(args.test_dir, args.output)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())