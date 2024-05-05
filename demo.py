import tkinter as tk
from tkinter import messagebox
import subprocess

def run_script():
    input_text = text_input.get("1.0", "end-1c")  # Get input text from text widget

    # Run the script with input text
    process = subprocess.Popen(['python', 'your_script.py', input_text], stdout=subprocess.PIPE)
    output, _ = process.communicate()

    # Show the output in a message box
    messagebox.showinfo("Output", output.decode('utf-8'))

# Create a Tkinter window
window = tk.Tk()
window.title("Script Runner")

# Create a text input widget
text_input = tk.Text(window, height=5, width=50)
text_input.pack()

# Create a button to run the script
run_button = tk.Button(window, text="Run Script", command=run_script)
run_button.pack()

# Run the Tkinter event loop
window.mainloop()