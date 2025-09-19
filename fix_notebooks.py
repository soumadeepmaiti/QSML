#!/usr/bin/env python3
"""
Script to fix common issues in all notebooks:
1. Import path corrections
2. Data loading fixes
3. Module reference corrections
"""

import json
import os
from pathlib import Path

def fix_notebook_imports_and_data_loading():
    """Fix imports and data loading in all notebooks"""
    
    notebooks_dir = Path("notebooks")
    
    # Standard data loading code to replace DataLoader usage
    data_loading_code = """# Load data directly from files
data_dir = Path('../data')

# Load equity data
equity_data = pd.read_csv(data_dir / 'raw' / 'equities' / 'combined_equities.csv')
equity_data['Date'] = pd.to_datetime(equity_data['Date'])

# Load treasury data  
treasury_data = pd.read_csv(data_dir / 'external' / 'treasury_yields.csv')
treasury_data['Date'] = pd.to_datetime(treasury_data['Date'])

# Load options data
options_data = pd.read_csv(data_dir / 'raw' / 'options' / 'spy_options.csv')
options_data['Date'] = pd.to_datetime(options_data['Date'])
options_data['Expiration'] = pd.to_datetime(options_data['Expiration'])

print(f"Loaded data:")
print(f"  - Equity: {len(equity_data)} rows")
print(f"  - Treasury: {len(treasury_data)} rows") 
print(f"  - Options: {len(options_data)} rows")"""
    
    # Process each notebook
    for notebook_file in notebooks_dir.glob("*.ipynb"):
        print(f"Processing {notebook_file.name}...")
        
        # Read notebook
        with open(notebook_file, 'r') as f:
            notebook = json.load(f)
        
        # Track if changes were made
        changes_made = False
        
        # Process each cell
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                
                # Fix import issues
                if 'from models.' in source or 'from data.data_loader import DataLoader' in source:
                    # Remove problematic imports
                    lines = cell['source']
                    new_lines = []
                    
                    for line in lines:
                        if ('from models.' in line or 
                            'from data.data_loader import DataLoader' in line):
                            continue
                        new_lines.append(line)
                    
                    cell['source'] = new_lines
                    changes_made = True
                
                # Fix data loading
                if 'data_loader = DataLoader' in source:
                    cell['source'] = [data_loading_code + '\n']
                    changes_made = True
        
        # Save if changes were made
        if changes_made:
            with open(notebook_file, 'w') as f:
                json.dump(notebook, f, indent=2)
            print(f"  ✅ Fixed {notebook_file.name}")
        else:
            print(f"  ⏭️  No changes needed for {notebook_file.name}")

if __name__ == "__main__":
    fix_notebook_imports_and_data_loading()
    print("All notebooks processed!")