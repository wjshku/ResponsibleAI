#!/usr/bin/env python3
"""
DANN Experiments Comparison Script

This script generates an HTML page comparing the performance and configuration
of different DANN (Domain Adversarial Neural Network) models.
"""

import json
from pathlib import Path
import base64


def get_model_info(model_dir):
    """Extract model information from metadata and evaluation results."""
    info = {
        'name': model_dir.name,
        'path': model_dir,
        'has_evaluation': False,
        'config': {},
        'source_metrics': {},
        'target_metrics': {},
        'training_info': {}
    }

    # Read metadata
    metadata_file = model_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            info['config'] = metadata.get('config', {})
            info['training_info'] = {
                'best_epoch': metadata.get('best_epoch'),
                'best_target_accuracy': metadata.get('best_target_accuracy')
            }

    # Read evaluation results
    eval_file = model_dir / "evaluation_results.json"
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            results = json.load(f)
            info['has_evaluation'] = True
            info['source_metrics'] = results.get('source', {})
            info['target_metrics'] = results.get('target', {})

    return info


def generate_html(models_info):
    """Generate HTML page with model comparison."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DANN Model Experiments Comparison</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .summary {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .summary h2 {{
            margin-top: 0;
            color: #34495e;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f4fd;
        }}
        .config-table {{
            font-size: 0.9em;
        }}
        .metrics-table {{
            font-size: 0.95em;
        }}
        .metric-good {{
            color: #27ae60;
            font-weight: bold;
        }}
        .metric-poor {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .metric-neutral {{
            color: #f39c12;
            font-weight: bold;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #34495e;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 5px;
        }}
        .model-name {{
            font-weight: bold;
            font-size: 1.1em;
        }}
        .best-model {{
            background-color: #d4edda !important;
            border-left: 4px solid #28a745;
        }}
        .comparison-section {{
            margin-bottom: 30px;
        }}
        .comparison-table {{
            border-collapse: collapse;
            width: 100%;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .comparison-table th {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 15px 10px;
            text-align: center;
            font-weight: 600;
            border: 1px solid #ddd;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        .comparison-table .metric-name {{
            background: #f8f9fa;
            font-weight: 600;
            text-align: left;
            min-width: 200px;
            border: 1px solid #ddd;
            padding: 12px 15px;
        }}
        .comparison-table .model-column {{
            background: #e3f2fd;
            font-weight: 600;
            text-align: center;
            min-width: 120px;
            border: 1px solid #ddd;
            padding: 12px 10px;
        }}
        .comparison-table .best-column {{
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border: 2px solid #28a745;
        }}
        .comparison-table td {{
            border: 1px solid #ddd;
            padding: 10px 8px;
            text-align: center;
            vertical-align: middle;
        }}
        .comparison-table .model-data {{
            background: #fafafa;
        }}
        .comparison-table .best-data {{
            background: #f0f8f0;
            font-weight: 600;
        }}
        .section-header td {{
            background: #34495e !important;
            color: white !important;
            font-weight: bold !important;
            font-size: 1.1em !important;
            text-align: center !important;
            padding: 15px !important;
            border: 2px solid #2c3e50 !important;
        }}
        .section-title {{
            font-size: 1.2em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .footer {{
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DANN Model Experiments Comparison</h1>

        <div class="summary">
            <h2>Summary</h2>
            <p>Comparing {len(models_info)} DANN models trained for domain adaptation between SD2 and Kontext datasets.</p>
            <p><strong>Goal:</strong> Achieve good performance on target domain (Kontext) while maintaining performance on source domain (SD2).</p>
            <p><strong>Key Metrics:</strong> Target domain accuracy, domain classification accuracy (~50% indicates good adaptation), and balanced source/target performance.</p>
        </div>

        <!-- Side-by-side model comparison table -->
        <div class="comparison-section">
            <h2>Model Comparison</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th class="metric-name">Metric</th>"""

    # Add model headers
    for info in models_info:
        if not info['has_evaluation']:
            continue
        full_name = info['name']
        short_name = full_name.replace('model_dann_', '').replace('_', ' ')
        is_best = info.get('is_best', False)
        best_badge = " 🏆" if is_best else ""
        html += f"""
                        <th class="model-column {'best-column' if is_best else ''}">{short_name}{best_badge}</th>"""

    html += """
                    </tr>
                </thead>
                <tbody>"""

    # Configuration section
    evaluated_models = [m for m in models_info if m.get('has_evaluation', False)]
    html += f"""
                    <tr class="section-header">
                        <td colspan="{len(evaluated_models) + 1}" class="section-title">Configuration</td>
                    </tr>"""

    config_metrics = [
        ('Batch Size', 'batch_size'),
        ('Learning Rate', 'learning_rate'),
        ('Epochs', 'num_epochs'),
        ('Feature Hidden Size', 'feature_hidden_size'),
        ('Domain Hidden Size', 'domain_hidden_size'),
        ('Dropout', 'dropout'),
        ('Sample Size', 'sample_size'),
        ('Gamma (λ)', 'gamma'),
        ('Zeta (η)', 'zeta'),
        ('Best Epoch', 'best_epoch', 'training_info'),
        ('Best Target Accuracy', 'best_target_accuracy', 'training_info')
    ]

    for metric_name, config_key, *source in config_metrics:
        html += f"""
                    <tr>
                        <td class="metric-name">{metric_name}</td>"""

        for info in evaluated_models:
            if source and source[0] == 'training_info':
                value = info['training_info'].get(config_key, 'N/A')
                if config_key == 'best_target_accuracy' and isinstance(value, (int, float)):
                    value = f"{value:.2f}%"
            else:
                config = info['config']
                value = config.get(config_key, 'N/A')

                # Handle special cases
                if config_key == 'sample_size':
                    if value == 'null' or value is None:
                        value = 'all'
                    elif isinstance(value, str) and value.startswith('all ('):
                        value = value  # Keep as-is
                elif config_key in ['gamma', 'zeta'] and value == 'N/A':
                    # Default values for older models
                    value = 10.0 if config_key == 'gamma' else 1.0

            html += f"""
                        <td class="model-data {'best-data' if info.get('is_best', False) else ''}">{value}</td>"""

        html += """
                    </tr>"""

    # Source Domain section
    html += f"""
                    <tr class="section-header">
                        <td colspan="{len(evaluated_models) + 1}" class="section-title">Source Domain Performance (SD2)</td>
                    </tr>"""

    performance_metrics = [
        ('Overall Accuracy', 'overall_accuracy'),
        ('Real Accuracy', 'real_accuracy'),
        ('Fake Accuracy', 'fake_accuracy'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1 Score', 'f1_score'),
        ('Domain Accuracy', 'domain_accuracy'),
        ('Total Samples', 'total_samples')
    ]

    for metric_name, metric_key in performance_metrics:
        html += f"""
                    <tr>
                        <td class="metric-name">{metric_name}</td>"""

        for info in evaluated_models:
            metrics = info['source_metrics']
            value = metrics.get(metric_key, 0)

            # Format based on metric type
            if metric_key in ['precision', 'recall', 'f1_score']:
                formatted_value = f"{value:.4f}"
            elif metric_key == 'total_samples':
                formatted_value = f"{int(value)}"
            elif metric_key == 'domain_accuracy':
                domain_class = "metric-good" if 40 <= value <= 60 else "metric-poor" if value > 70 else "metric-neutral"
                formatted_value = f'<span class="{domain_class}">{value:.2f}%</span>'
            else:
                formatted_value = f"{value:.2f}%"

            html += f"""
                        <td class="model-data {'best-data' if info.get('is_best', False) else ''}">{formatted_value}</td>"""

        html += """
                    </tr>"""

    # Target Domain section
    html += f"""
                    <tr class="section-header">
                        <td colspan="{len(evaluated_models) + 1}" class="section-title">Target Domain Performance (Kontext)</td>
                    </tr>"""

    for metric_name, metric_key in performance_metrics:  # Same metrics as source
        html += f"""
                    <tr>
                        <td class="metric-name">{metric_name}</td>"""

        for info in evaluated_models:
            metrics = info['target_metrics']
            value = metrics.get(metric_key, 0)

            # Format based on metric type
            if metric_key in ['precision', 'recall', 'f1_score']:
                formatted_value = f"{value:.4f}"
            elif metric_key == 'total_samples':
                formatted_value = f"{int(value)}"
            elif metric_key == 'domain_accuracy':
                domain_class = "metric-good" if 40 <= value <= 60 else "metric-poor" if value > 70 else "metric-neutral"
                formatted_value = f'<span class="{domain_class}">{value:.2f}%</span>'
            else:
                formatted_value = f"{value:.2f}%"

            html += f"""
                        <td class="model-data {'best-data' if info.get('is_best', False) else ''}">{formatted_value}</td>"""

        html += """
                    </tr>"""

    # Domain Adaptation Analysis section
    html += f"""
                    <tr class="section-header">
                        <td colspan="{len(evaluated_models) + 1}" class="section-title">Domain Adaptation Analysis</td>
                    </tr>"""

    adaptation_metrics = [
        ('Source Domain Accuracy', 'domain_accuracy', 'source_metrics'),
        ('Target Domain Accuracy', 'domain_accuracy', 'target_metrics'),
        ('Domain Accuracy Gap', 'gap'),
        ('Assessment', 'assessment')
    ]

    for metric_name, *keys in adaptation_metrics:
        html += f"""
                    <tr>
                        <td class="metric-name">{metric_name}</td>"""

        for info in evaluated_models:
            if keys[0] == 'gap':
                source_domain_acc = info['source_metrics'].get('domain_accuracy', 0)
                target_domain_acc = info['target_metrics'].get('domain_accuracy', 0)
                value = abs(source_domain_acc - target_domain_acc)
                formatted_value = f"{value:.2f}%"
            elif keys[0] == 'assessment':
                source_domain_acc = info['source_metrics'].get('domain_accuracy', 0)
                target_domain_acc = info['target_metrics'].get('domain_accuracy', 0)
                gap = abs(source_domain_acc - target_domain_acc)

                if 40 <= source_domain_acc <= 60 and 40 <= target_domain_acc <= 60:
                    assessment = "Excellent"
                    assessment_class = "metric-good"
                elif gap < 20:
                    assessment = "Good"
                    assessment_class = "metric-good"
                elif source_domain_acc > 70 and target_domain_acc > 70:
                    assessment = "Poor"
                    assessment_class = "metric-poor"
                else:
                    assessment = "Moderate"
                    assessment_class = "metric-neutral"

                formatted_value = f'<span class="{assessment_class}">{assessment}</span>'
            else:
                metrics = info[keys[1]]
                value = metrics.get(keys[0], 0)
                domain_class = "metric-good" if 40 <= value <= 60 else "metric-poor" if value > 70 else "metric-neutral"
                formatted_value = f'<span class="{domain_class}">{value:.2f}%</span>'

            html += f"""
                        <td class="model-data {'best-data' if info.get('is_best', False) else ''}">{formatted_value}</td>"""

        html += """
                    </tr>"""

    html += """
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Generated by DANN Experiments Comparison Script</p>
            <p>Domain Adversarial Neural Network (DANN) for SD2 to Kontext adaptation</p>
        </div>
    </div>
</body>
</html>"""

    return html


def main():
    """Main function to generate experiments comparison."""
    print("DANN Experiments Comparison")
    print("=" * 50)

    # Find all DANN models
    models_dir = Path("models")
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        return

    dann_models = sorted([d for d in models_dir.iterdir()
                         if d.is_dir() and d.name.startswith("model_dann_")],
                        key=lambda x: x.stat().st_mtime, reverse=True)

    if not dann_models:
        print("No DANN models found")
        return

    print(f"Found {len(dann_models)} DANN models")

    # Collect model information
    models_info = []
    for model_dir in dann_models:
        print(f"Processing {model_dir.name}...")
        info = get_model_info(model_dir)
        if info['has_evaluation']:
            models_info.append(info)

    if not models_info:
        print("No models with evaluation results found")
        return

    # Sort by target domain accuracy (best first)
    models_info.sort(key=lambda x: x['target_metrics'].get('overall_accuracy', 0), reverse=True)
    if models_info:
        models_info[0]['is_best'] = True  # Mark the best performing model

    print(f"Generating comparison for {len(models_info)} evaluated models")

    # Generate HTML
    html_content = generate_html(models_info)

    # Write HTML file
    output_file = Path("dann_experiments_comparison.html")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Comparison page generated: {output_file}")
    print(f"Open {output_file} in your web browser to view the results")

    # Print summary to console
    print("\nModel Performance Summary:")
    print("-" * 50)
    for i, info in enumerate(models_info[:5]):  # Show top 5
        marker = " ⭐ BEST" if info.get('is_best', False) else ""
        target_acc = info['target_metrics'].get('overall_accuracy', 0)
        source_acc = info['source_metrics'].get('overall_accuracy', 0)
        print(f"{i+1}. {info['name']}{marker}")
        print(f"   Target: {target_acc:.1f}%, Source: {source_acc:.1f}%")


if __name__ == "__main__":
    main()
