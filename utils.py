import json
import numpy as np

def format_detection_output(staff_lines, notes):
    """Format detection results into JSON."""
    output = {
        "staff_lines": [
            {"y_position": int(y)} for y in staff_lines
        ],
        "notes": [
            {
                "pitch": note["pitch"],
                "duration": note["duration"],
                "position": {
                    "x": int(note["bbox"][0]),
                    "y": int(note["bbox"][1])
                }
            } for note in notes
        ]
    }
    return json.dumps(output, indent=2)

def get_note_pitch(y_position, staff_lines):
    """Calculate note pitch based on its position relative to staff lines."""
    staff_spacing = np.mean(np.diff(staff_lines))
    reference_line = staff_lines[2]  # Middle line
    distance = (reference_line - y_position) / (staff_spacing/2)
    
    # Map distance to pitch (simplified)
    pitches = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
    pitch_idx = int(distance + 4)
    pitch_idx = max(0, min(pitch_idx, len(pitches)-1))
    return pitches[pitch_idx]
    