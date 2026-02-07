"""
PHASE 4: VISUAL ALERT INTERFACE
==============================

Real-time vehicle diagnostic dashboard and alerts
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns

# ============================================================================
# ALERT INTERFACE
# ============================================================================

class AlertDisplay:
    """Generate visual alert displays"""
    
    @staticmethod
    def generate_alert_screen(diagnostic_report, output_path='reports/alert_display.png'):
        """
        Generate alert screen for vehicle display
        """
        
        status = diagnostic_report['vehicle_status']
        fault_name = diagnostic_report['detected_fault']
        confidence = diagnostic_report['confidence_score']
        color = diagnostic_report['display_color']
        action = diagnostic_report['immediate_action']
        
        # Color mapping
        color_rgb = {
            'green': '#2ecc71',
            'yellow': '#f39c12',
            'red': '#e74c3c'
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a1a')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Status bar (top)
        status_color = color_rgb.get(color, '#95a5a6')
        status_rect = patches.FancyBboxPatch(
            (0.2, 8.5), 9.6, 1.2, boxstyle="round,pad=0.1",
            edgecolor=status_color, facecolor=status_color, linewidth=3
        )
        ax.add_patch(status_rect)
        
        # Status text
        ax.text(5, 9.1, status, fontsize=32, weight='bold', color='white',
                ha='center', va='center', family='monospace')
        
        # Fault name (main alert)
        ax.text(5, 7.5, fault_name, fontsize=24, weight='bold',
                color=status_color, ha='center', va='center')
        
        # Confidence gauge
        ax.text(5, 6.5, f'Confidence: {confidence}%', fontsize=16, weight='bold',
                color='white', ha='center', va='center')
        
        # Confidence bar
        bar_width = (confidence / 100) * 8
        confidence_rect = patches.FancyBboxPatch(
            (1, 6.0), bar_width, 0.3, boxstyle="round,pad=0.02",
            edgecolor=status_color, facecolor=status_color, linewidth=2
        )
        ax.add_patch(confidence_rect)
        ax.plot([1, 9], [6.0, 6.0], 'w-', linewidth=1, alpha=0.3)
        
        # Divider
        ax.plot([0.5, 9.5], [5.5, 5.5], color=status_color, linewidth=2, alpha=0.5)
        
        # Action required
        ax.text(5, 4.8, 'ACTION REQUIRED:', fontsize=14, weight='bold',
                color='white', ha='center', va='top')
        
        # Action text (wrapped)
        action_text = '\n'.join(
            [action[i:i+40] for i in range(0, len(action), 40)]
        )
        ax.text(5, 4.2, action_text, fontsize=12, color='white',
                ha='center', va='top', wrap=True, family='monospace')
        
        # Urgency indicator
        urgency = diagnostic_report['urgency_level'].upper()
        urgency_colors = {
            'IMMEDIATE': '#c0392b',
            'URGENT': '#e74c3c',
            'SOON': '#f39c12',
            'MONITOR': '#3498db'
        }
        urgency_color = urgency_colors.get(urgency, '#95a5a6')
        
        urgency_rect = patches.FancyBboxPatch(
            (0.5, 0.5), 9, 0.8, boxstyle="round,pad=0.05",
            edgecolor=urgency_color, facecolor=urgency_color, linewidth=2, alpha=0.8
        )
        ax.add_patch(urgency_rect)
        
        ax.text(5, 0.9, f'⚠️  URGENCY: {urgency}', fontsize=12, weight='bold',
                color='white', ha='center', va='center', family='monospace')
        
        # Timestamp
        ax.text(5, 0.15, diagnostic_report['timestamp'][:10], fontsize=8,
                color='#95a5a6', ha='center', va='center', style='italic')
        
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"✅ Alert display saved to: {output_path}")
        plt.close()
        
        return output_path

# ============================================================================
# DIAGNOSTIC DASHBOARD
# ============================================================================

class DiagnosticDashboard:
    """Generate comprehensive diagnostic dashboard"""
    
    def __init__(self, history=None):
        """
        Initialize dashboard
        
        Args:
            history: List of diagnostic reports over time
        """
        self.history = history or []
    
    def add_report(self, report):
        """Add diagnostic report to history"""
        self.history.append(report)
    
    def generate_dashboard(self, output_path='reports/diagnostic_dashboard.png'):
        """
        Generate comprehensive dashboard visualization
        """
        
        if not self.history:
            print("⚠️  No diagnostic history available")
            return None
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        fig.suptitle('HearAI-EV Diagnostic Dashboard', fontsize=20, weight='bold', y=0.98)
        
        # Extract data from history
        statuses = [r['vehicle_status'] for r in self.history]
        confidences = [r['confidence_score'] for r in self.history]
        severity_levels = [r['severity_numeric'] for r in self.history]
        faults = [r['detected_fault'] for r in self.history]
        timestamps = [r['timestamp'] for r in self.history]
        
        # 1. Timeline of status changes
        ax1 = fig.add_subplot(gs[0, :2])
        colors = ['#2ecc71' if s == 'HEALTHY' else '#f39c12' if s == 'WARNING' else '#e74c3c'
                  for s in statuses]
        ax1.scatter(range(len(statuses)), severity_levels, c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)
        ax1.plot(range(len(statuses)), severity_levels, 'k--', alpha=0.3, linewidth=1)
        ax1.set_xlabel('Sample Number')
        ax1.set_ylabel('Severity Level')
        ax1.set_title('Vehicle Status Timeline', fontweight='bold')
        ax1.set_ylim([0, 10])
        ax1.grid(alpha=0.3)
        
        # 2. Current status summary
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        latest = self.history[-1]
        summary_text = f"""
LATEST STATUS
{'─'*20}
Status: {latest['vehicle_status']}
Fault: {latest['detected_fault']}
Confidence: {latest['confidence_score']}%
Severity: {latest['severity_text']}
Urgency: {latest['urgency_level']}
        """
        ax2.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Confidence trend
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(range(len(confidences)), confidences, 'o-', color='#3498db', linewidth=2, markersize=6)
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Confidence Threshold')
        ax3.fill_between(range(len(confidences)), confidences, 70, alpha=0.2, color='#3498db')
        ax3.set_ylabel('Confidence (%)')
        ax3.set_title('Confidence Score Trend', fontweight='bold')
        ax3.set_ylim([0, 100])
        ax3.grid(alpha=0.3)
        ax3.legend()
        
        # 4. Fault type distribution
        ax4 = fig.add_subplot(gs[1, 1])
        fault_counts = {}
        for fault in faults:
            fault_type = fault.split('-')[0].strip()
            fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1
        
        colors_pie = ['#e74c3c' if 'bearing' in ft.lower() else '#f39c12' if 'propeller' in ft.lower() else '#2ecc71'
                      for ft in fault_counts.keys()]
        ax4.pie(fault_counts.values(), labels=fault_counts.keys(), autopct='%1.1f%%',
                colors=colors_pie, startangle=90)
        ax4.set_title('Fault Type Distribution', fontweight='bold')
        
        # 5. Severity distribution
        ax5 = fig.add_subplot(gs[1, 2])
        severity_dist = {}
        for r in self.history:
            sev = r['severity_text']
            severity_dist[sev] = severity_dist.get(sev, 0) + 1
        
        sev_order = ['none', 'low', 'medium', 'high']
        sev_counts = [severity_dist.get(s, 0) for s in sev_order]
        colors_sev = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
        
        ax5.barh(sev_order, sev_counts, color=colors_sev)
        ax5.set_xlabel('Count')
        ax5.set_title('Severity Distribution', fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)
        
        # 6. Last 10 detections
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        last_reports = self.history[-10:] if len(self.history) >= 10 else self.history
        table_data = []
        
        for i, report in enumerate(last_reports):
            table_data.append([
                i + 1,
                report['vehicle_status'],
                report['detected_fault'][:25],
                f"{report['confidence_score']}%",
                report['severity_text'],
                report['timestamp'][-8:]  # HH:MM:SS
            ])
        
        table = ax6.table(cellText=table_data,
                         colLabels=['#', 'Status', 'Fault', 'Conf', 'Severity', 'Time'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.05, 0.15, 0.35, 0.1, 0.15, 0.1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color code rows
        for i in range(1, len(table_data) + 1):
            status = table_data[i-1][1]
            color = '#e8f8f5' if status == 'HEALTHY' else '#fef5e7' if status == 'WARNING' else '#fadbd8'
            for j in range(6):
                table[(i, j)].set_facecolor(color)
        
        # Header
        for j in range(6):
            table[(0, j)].set_facecolor('#34495e')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('Recent Diagnostic History', fontweight='bold', loc='left', pad=20)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Dashboard saved to: {output_path}")
        plt.close()
        
        return output_path
    
    def generate_html_dashboard(self, output_path='reports/dashboard.html'):
        """Generate interactive HTML dashboard"""
        
        if not self.history:
            print("⚠️  No diagnostic history available")
            return None
        
        # Prepare data
        latest = self.history[-1]
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HearAI-EV Diagnostic Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        h1 {{
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .status-card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            border-left: 5px solid #{'2ecc71' if latest['vehicle_status'] == 'HEALTHY' else 'f39c12' if latest['vehicle_status'] == 'WARNING' else 'e74c3c'};
        }}
        
        .status-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .status-title {{
            font-size: 2em;
            font-weight: bold;
            color: #{'2ecc71' if latest['vehicle_status'] == 'HEALTHY' else 'f39c12' if latest['vehicle_status'] == 'WARNING' else 'e74c3c'};
        }}
        
        .confidence-score {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 20px;
        }}
        
        .info-item {{
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        
        .info-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
            font-weight: 600;
        }}
        
        .info-value {{
            font-size: 1.2em;
            color: #333;
            font-weight: bold;
        }}
        
        .message {{
            background: #f0f7ff;
            border-left: 4px solid #3498db;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            color: #333;
            line-height: 1.6;
        }}
        
        .action {{
            background: #fff3cd;
            border-left: 4px solid #f39c12;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            color: #333;
            line-height: 1.6;
        }}
        
        .recommendations {{
            margin-top: 20px;
        }}
        
        .recommendations h3 {{
            color: #333;
            margin-bottom: 10px;
        }}
        
        .recommendations ul {{
            list-style-position: inside;
            color: #666;
            line-height: 2;
        }}
        
        .history-table {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        
        tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        .status-healthy {{ color: #2ecc71; font-weight: bold; }}
        .status-warning {{ color: #f39c12; font-weight: bold; }}
        .status-critical {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 HearAI-EV Diagnostic Dashboard</h1>
        
        <div class="status-card">
            <div class="status-header">
                <div class="status-title">
                    {'✅' if latest['vehicle_status'] == 'HEALTHY' else '⚠️' if latest['vehicle_status'] == 'WARNING' else '❌'} {latest['vehicle_status']}
                </div>
                <div class="confidence-score">{latest['confidence_score']}% Confidence</div>
            </div>
            
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Detected Issue</div>
                    <div class="info-value">{latest['detected_fault']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Severity Level</div>
                    <div class="info-value">{latest['severity_text'].upper()}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Urgency</div>
                    <div class="info-value">{latest['urgency_level'].upper()}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Last Updated</div>
                    <div class="info-value">{latest['timestamp']}</div>
                </div>
            </div>
            
            <div class="message">
                <strong>Diagnostic Message:</strong><br>
                {latest['diagnostic_explanation']}
            </div>
            
            <div class="action">
                <strong>⚡ Immediate Action:</strong><br>
                {latest['immediate_action']}
            </div>
            
            <div class="recommendations">
                <h3>📋 Recommendations:</h3>
                <ul>
                    {''.join([f'<li>{rec}</li>' for rec in latest['recommendations']])}
                </ul>
            </div>
        </div>
        
        <div class="history-table">
            <h2>📊 Recent Diagnostic History</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Status</th>
                        <th>Detected Fault</th>
                        <th>Confidence</th>
                        <th>Severity</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join([f'''
                    <tr>
                        <td>{i+1}</td>
                        <td class="status-{r['vehicle_status'].lower()}">{r['vehicle_status']}</td>
                        <td>{r['detected_fault'][:40]}</td>
                        <td>{r['confidence_score']}%</td>
                        <td>{r['severity_text'].upper()}</td>
                        <td>{r['timestamp'][-8:]}</td>
                    </tr>
                    ''' for i, r in enumerate(self.history[-20:])])}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
        """
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"✅ HTML dashboard saved to: {output_path}")
        return output_path

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_interface():
    """Generate example interface"""
    
    print("\n" + "="*70)
    print("GENERATING ALERT INTERFACE EXAMPLES")
    print("="*70)
    
    # Example diagnostic report
    example_report = {
        'timestamp': datetime.now().isoformat(),
        'vehicle_status': 'WARNING',
        'detected_fault': 'BEARING WEAR - WARNING',
        'confidence_score': 82.5,
        'severity_numeric': 6,
        'severity_text': 'medium',
        'model_probabilities': {
            'bearing': 0.825,
            'healthy': 0.12,
            'propeller': 0.055
        },
        'diagnostic_explanation': 'Moderate bearing wear detected. The acoustic signature indicates early-stage degradation with ball/roller surface irregularities.',
        'symptoms_detected': ['Persistent humming', 'Slight vibration increase', 'Temperature rise'],
        'immediate_action': 'Schedule maintenance within 24-48 hours. Monitor closely during operation.',
        'recommendations': ['Schedule maintenance within 24-48 hours', 'Reduce driving stress on the component'],
        'maintenance_guide': [
            'Verify unusual sounds in quiet environment',
            'Check for vibration patterns during acceleration',
            'Monitor bearing temperature if accessible'
        ],
        'urgency_level': 'urgent',
        'display_color': 'yellow'
    }
    
    # Generate alert display
    AlertDisplay.generate_alert_screen(example_report)
    
    # Generate dashboard with history
    dashboard = DiagnosticDashboard()
    
    # Add multiple reports to simulate history
    for i in range(5):
        report = example_report.copy()
        report['confidence_score'] = 70 + i * 3
        report['severity_numeric'] = 4 + i
        dashboard.add_report(report)
    
    dashboard.generate_dashboard()
    dashboard.generate_html_dashboard()
    
    print("\n✅ All interface examples generated!")

if __name__ == "__main__":
    example_interface()
