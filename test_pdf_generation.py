#!/usr/bin/env python3
"""
Test PDF Generation
Simple test to verify PDF creation is working
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def create_test_pdf():
    """Create a simple test PDF"""
    print("📄 Creating test PDF...")
    
    with PdfPages('test_pdf_generation.pdf') as pdf:
        # Create a simple figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Simple bar chart
        categories = ['A', 'B', 'C', 'D']
        values = [10, 20, 15, 25]
        ax1.bar(categories, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax1.set_title('Test Bar Chart', fontweight='bold')
        ax1.set_ylabel('Values')
        
        # Simple line plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax2.plot(x, y, color='#3498db', linewidth=2)
        ax2.set_title('Test Line Plot', fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print("✅ Test PDF created: test_pdf_generation.pdf")

if __name__ == "__main__":
    create_test_pdf() 