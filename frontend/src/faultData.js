export const data = {
  healthy: {
    label: "Healthy",
    color: "#00e676",
    rows: [
      {
        class: "Healthy - Excellent",
        subtype: "Optimal Operation",
        keyReason: "No faults, clean signal",
        signalFeature: "Steady baseline",
        severity: "None",
        nextService: "Routine check at 10,000 km",
        notes: "No action needed."
      },
      {
        class: "Healthy - Degrading",
        subtype: "Pre-fault State",
        keyReason: "Accelerated wear onset",
        signalFeature: "Subtle frequency anomalies",
        severity: "Watch",
        nextService: "Service within 500 km",
        notes: "Preventive inspection advised."
      }
    ]
  },
  bearing: {
    label: "Bearing Faults",
    color: "#ff6d00",
    rows: [
      {
        class: "Bearing - Outer Race",
        subtype: "Outer Ring Defect (BPFO)",
        keyReason: "Surface pitting on outer ring",
        signalFeature: "Impulse spikes at BPFO",
        severity: "Moderate -> High",
        nextService: "Replace bearing within 200 km",
        notes: "Common bearing fault."
      },
      {
        class: "Bearing - Inner Race",
        subtype: "Inner Ring Defect (BPFI)",
        keyReason: "Rotating defect on inner ring",
        signalFeature: "Amplitude-modulated impulses",
        severity: "High",
        nextService: "Replace within 100 km",
        notes: "Needs fast inspection."
      }
    ]
  },
  propeller: {
    label: "Propeller Faults",
    color: "#2979ff",
    rows: [
      {
        class: "Propeller - Blade Imbalance",
        subtype: "Uneven Blade Mass",
        keyReason: "Debris or manufacturing variation",
        signalFeature: "Strong 1P vibration",
        severity: "Moderate",
        nextService: "Dynamic balancing within 300 km",
        notes: "Most common propeller fault."
      },
      {
        class: "Propeller - Blade Crack",
        subtype: "Structural Fatigue Crack",
        keyReason: "Fatigue or impact damage",
        signalFeature: "Random spikes and distortion",
        severity: "Critical",
        nextService: "Immediate shutdown",
        notes: "Safety critical."
      }
    ]
  }
};

export const cols = [
  { key: "class", label: "Class" },
  { key: "subtype", label: "Subtype" },
  { key: "keyReason", label: "Key Reason" },
  { key: "signalFeature", label: "Signal Feature" },
  { key: "severity", label: "Severity" },
  { key: "nextService", label: "Next Service" },
  { key: "notes", label: "Notes" }
];
