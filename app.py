import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import soundfile as sf
import librosa
import threading
import onnxruntime as ort
from audio_transcriber import TDTTranscriber, CTCTranscriber
import torch
import psutil
import os
import gc

class AudioTranscriberApp:
    """Audio transcriber with CUDA and timestamp support."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcriber")
        self.root.geometry("800x800")
        
        self.transcriber = None
        self.target_sample_rate = 16000
        self.is_working = False
        self.use_cuda = self.check_cuda_available()
        
        # Memory monitoring settings
        self.memory_threshold = 0.80  # 80% of available memory
        self.vram_threshold = 0.80    # 80% of available VRAM if using CUDA
        self.default_chunk_duration = 600  # Default 10 minutes in seconds
        
        self.create_widgets()
        self.setup_menu()
        
    def check_cuda_available(self):
        """Check if CUDA is available and print detailed info."""
        try:
            # Check ONNX providers
            providers = ort.get_available_providers()
            is_cuda_available = "CUDAExecutionProvider" in providers
            
            print("ONNX Providers:", providers)
            
            # Check PyTorch CUDA availability
            if is_cuda_available:
                pytorch_cuda = torch.cuda.is_available()
                is_cuda_available = pytorch_cuda
                
                print(f"PyTorch CUDA available: {pytorch_cuda}")
                
                if pytorch_cuda:
                    # Print detailed CUDA info
                    device_count = torch.cuda.device_count()
                    print(f"CUDA device count: {device_count}")
                    
                    for i in range(device_count):
                        device_name = torch.cuda.get_device_name(i)
                        device_cap = torch.cuda.get_device_capability(i)
                        print(f"CUDA Device {i}: {device_name}, Capability: {device_cap}")
                        
                        # Try to get memory info
                        try:
                            total_mem = torch.cuda.get_device_properties(i).total_memory
                            allocated_mem = torch.cuda.memory_allocated(i)
                            reserved_mem = torch.cuda.memory_reserved(i)
                            
                            print(f"  Total memory: {total_mem / 1024**2:.2f} MB")
                            print(f"  Allocated memory: {allocated_mem / 1024**2:.2f} MB")
                            print(f"  Reserved memory: {reserved_mem / 1024**2:.2f} MB")
                            print(f"  Memory utilization: {(allocated_mem / total_mem) * 100:.2f}%")
                        except Exception as e:
                            print(f"  Error getting memory info: {e}")
                    
                    # Force some CUDA allocation to test memory tracking
                    try:
                        x = torch.zeros(1000, 1000).cuda()  # Allocate a small tensor
                        allocated_after = torch.cuda.memory_allocated(0)
                        print(f"After test allocation: {allocated_after / 1024**2:.2f} MB")
                        del x  # Clean up
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Test allocation error: {e}")
            
            return is_cuda_available
        except Exception as e:
            print(f"CUDA check error: {e}")
            return False
    
    def setup_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.browse_file)
        file_menu.add_command(label="Save", command=self.save_transcription)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Advanced menu
        advanced_menu = tk.Menu(menubar, tearoff=0)
        advanced_menu.add_command(label="Memory Settings", command=self.show_memory_settings)
        menubar.add_cascade(label="Advanced", menu=advanced_menu)
        
        self.root.config(menu=menubar)
    
    def show_memory_settings(self):
        """Show dialog for memory settings."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Memory Settings")
        settings_window.geometry("640x640")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        ttk.Label(settings_window, text="Memory Threshold (%)").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        memory_scale = ttk.Scale(
            settings_window, 
            from_=50, 
            to=95, 
            orient=tk.HORIZONTAL, 
            value=self.memory_threshold * 100,
            length=200
        )
        memory_scale.grid(row=0, column=1, padx=10, pady=10)
        memory_value = ttk.Label(settings_window, text=f"{int(self.memory_threshold * 100)}%")
        memory_value.grid(row=0, column=2, padx=10, pady=10)
        
        def update_memory_value(event):
            value = int(memory_scale.get())
            memory_value.config(text=f"{value}%")
        
        memory_scale.bind("<Motion>", update_memory_value)
        
        # VRAM settings
        ttk.Label(settings_window, text="VRAM Threshold (%)").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        vram_scale = ttk.Scale(
            settings_window, 
            from_=50, 
            to=95, 
            orient=tk.HORIZONTAL, 
            value=self.vram_threshold * 100,
            length=200
        )
        vram_scale.grid(row=1, column=1, padx=10, pady=10)
        vram_value = ttk.Label(settings_window, text=f"{int(self.vram_threshold * 100)}%")
        vram_value.grid(row=1, column=2, padx=10, pady=10)
        
        def update_vram_value(event):
            value = int(vram_scale.get())
            vram_value.config(text=f"{value}%")
        
        vram_scale.bind("<Motion>", update_vram_value)
        
        # Chunk size settings
        ttk.Label(settings_window, text="Default Chunk Size (seconds)").grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        chunk_entry = ttk.Entry(settings_window)
        chunk_entry.insert(0, str(self.default_chunk_duration))
        chunk_entry.grid(row=2, column=1, padx=10, pady=10)
        
        # System information
        info_frame = ttk.LabelFrame(settings_window, text="System Information")
        info_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky=tk.W+tk.E)
        
        # Current memory usage labels (will be updated)
        mem_info_frame = ttk.Frame(info_frame)
        mem_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ram_label = ttk.Label(mem_info_frame, text="RAM: Checking...")
        ram_label.pack(anchor=tk.W, padx=10, pady=2)
        
        vram_label = ttk.Label(mem_info_frame, text="VRAM: Checking...")
        vram_label.pack(anchor=tk.W, padx=10, pady=2)
        
        # Function to update memory info in real-time
        def update_memory_info():
            """Update memory info display with complete information."""
            mem_info = self.get_memory_usage()
            
            # Update RAM info
            ram_total = mem_info['ram_total'] / (1024**3)
            ram_available = mem_info['ram_available'] / (1024**3)
            ram_percent = mem_info['ram_percent'] * 100
            ram_label.config(text=f"RAM: {ram_percent:.1f}% used, {ram_available:.2f} GB free of {ram_total:.2f} GB")
            
            # Update VRAM info if available
            if self.use_cuda and torch.cuda.is_available():
                try:
                    vram_total = mem_info.get('vram_total', 0) / (1024**3)
                    vram_allocated = mem_info.get('vram_allocated', 0) / (1024**3)
                    vram_reserved = mem_info.get('vram_reserved', 0) / (1024**3)
                    vram_percent = mem_info.get('vram_percent', 0) * 100
                    
                    # Base display with PyTorch info
                    vram_text = f"VRAM: {vram_percent:.1f}% used, "
                    
                    # If we have nvidia-smi data, show both sources
                    if 'vram_nvidia_used' in mem_info:
                        nvidia_used = mem_info.get('vram_nvidia_used', 0) / (1024**3)
                        nvidia_total = mem_info.get('vram_nvidia_total', 0) / (1024**3)
                        vram_text += f"{nvidia_used:.2f} GB used (nvidia-smi) / {nvidia_total:.2f} GB total"
                        vram_text += f"\nPyTorch: {vram_allocated:.2f} GB allocated, {vram_reserved:.2f} GB reserved of {vram_total:.2f} GB"
                    else:
                        vram_text += f"{vram_allocated:.2f} GB allocated, {vram_reserved:.2f} GB reserved of {vram_total:.2f} GB"
                    
                    vram_label.config(text=vram_text)
                except Exception as e:
                    vram_label.config(text=f"VRAM: Error getting info - {str(e)}")
            else:
                vram_label.config(text="VRAM: CUDA not available")
        
        # Button to refresh memory info
        ttk.Button(info_frame, text="Refresh Memory Info", command=update_memory_info).pack(anchor=tk.W, padx=10, pady=5)
        
        # Debug button for CUDA info
        def show_cuda_info():
            """Enhanced CUDA debug info display."""
            debug_window = tk.Toplevel(self.root)
            debug_window.title("CUDA Debug Info")
            debug_window.geometry("700x500")
            
            debug_text = tk.Text(debug_window, wrap=tk.WORD)
            scrollbar = ttk.Scrollbar(debug_window, command=debug_text.yview)
            debug_text.config(yscrollcommand=scrollbar.set)
            debug_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Redirect print statements to the text widget
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            
            # Basic PyTorch CUDA info
            debug_text.insert(tk.END, "===== PyTorch CUDA Information =====\n")
            debug_text.insert(tk.END, f"CUDA available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                debug_text.insert(tk.END, f"CUDA current device: {torch.cuda.current_device()}\n")
                debug_text.insert(tk.END, f"CUDA device count: {torch.cuda.device_count()}\n")
                
                for i in range(torch.cuda.device_count()):
                    debug_text.insert(tk.END, f"\nDevice {i}: {torch.cuda.get_device_name(i)}\n")
                    debug_text.insert(tk.END, f"  CUDA capability: {torch.cuda.get_device_capability(i)}\n")
                    
                    # Memory info
                    props = torch.cuda.get_device_properties(i)
                    total_mem = props.total_memory
                    debug_text.insert(tk.END, f"  Total memory: {total_mem / 1024**2:.2f} MB ({total_mem / 1024**3:.2f} GB)\n")
                    
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    debug_text.insert(tk.END, f"  Memory allocated: {allocated / 1024**2:.2f} MB ({allocated / 1024**3:.2f} GB)\n")
                    debug_text.insert(tk.END, f"  Memory reserved: {reserved / 1024**2:.2f} MB ({reserved / 1024**3:.2f} GB)\n")
                    
                    if allocated > 0:
                        debug_text.insert(tk.END, f"  Allocation % of total: {(allocated / total_mem) * 100:.2f}%\n")
                    if reserved > 0:
                        debug_text.insert(tk.END, f"  Reserved % of total: {(reserved / total_mem) * 100:.2f}%\n")
            
            # Try to get nvidia-smi info
            debug_text.insert(tk.END, "\n===== NVIDIA-SMI Information =====\n")
            try:
                import subprocess
                result = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
                debug_text.insert(tk.END, result)
            except Exception as e:
                debug_text.insert(tk.END, f"nvidia-smi not available: {str(e)}\n")
            
            # Force some CUDA allocation to test
            debug_text.insert(tk.END, "\n===== CUDA Memory Allocation Test =====\n")
            if torch.cuda.is_available():
                try:
                    debug_text.insert(tk.END, "Allocating test tensors...\n")
                    before = torch.cuda.memory_allocated(0)
                    debug_text.insert(tk.END, f"Before allocation: {before / 1024**2:.2f} MB\n")
                    
                    # Allocate a 1GB tensor
                    x = torch.zeros(1024, 1024, 256, dtype=torch.float32, device='cuda')
                    after = torch.cuda.memory_allocated(0)
                    debug_text.insert(tk.END, f"After allocation: {after / 1024**2:.2f} MB\n")
                    debug_text.insert(tk.END, f"Difference: {(after - before) / 1024**2:.2f} MB\n")
                    
                    # Clean up
                    del x
                    torch.cuda.empty_cache()
                    cleaned = torch.cuda.memory_allocated(0)
                    debug_text.insert(tk.END, f"After cleanup: {cleaned / 1024**2:.2f} MB\n")
                except Exception as e:
                    debug_text.insert(tk.END, f"Allocation test error: {str(e)}\n")
            
            # Get the printed output
            sys.stdout = old_stdout
            debug_text.insert(tk.END, "\n===== Debug Output =====\n")
            debug_text.insert(tk.END, mystdout.getvalue())
        
        if self.use_cuda:
            ttk.Button(info_frame, text="Debug CUDA Info", command=show_cuda_info).pack(anchor=tk.W, padx=10, pady=5)
        
        # Run initial memory update
        update_memory_info()
        
        def save_settings():
            try:
                self.memory_threshold = int(memory_scale.get()) / 100
                self.vram_threshold = int(vram_scale.get()) / 100
                self.default_chunk_duration = int(chunk_entry.get())
                settings_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers")
        
        ttk.Button(settings_window, text="Save", command=save_settings).grid(row=4, column=1, pady=20)
        
    def create_widgets(self):
        """Create GUI components."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Audio File", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=60).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Model selection
        model_frame = ttk.Frame(settings_frame)
        model_frame.pack(fill=tk.X, pady=5)
        
        self.model_var = tk.StringVar(value="tdt")
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="TDT", variable=self.model_var, value="tdt", command=self.update_timestamp_availability).pack(side=tk.LEFT)
        ttk.Radiobutton(model_frame, text="CTC", variable=self.model_var, value="ctc", command=self.update_timestamp_availability).pack(side=tk.LEFT)
        
        format_frame = ttk.Frame(settings_frame)
        format_frame.pack(fill=tk.X, pady=5)

        self.output_format = tk.StringVar(value="text")
        ttk.Label(format_frame, text="Output Format:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="Text", variable=self.output_format, value="text").pack(side=tk.LEFT)
        ttk.Radiobutton(format_frame, text="SRT", variable=self.output_format, value="srt").pack(side=tk.LEFT)
        
        # Memory management checkbox
        memory_frame = ttk.Frame(settings_frame)
        memory_frame.pack(fill=tk.X, pady=5)
        
        self.auto_chunk_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            memory_frame, 
            text="Auto-Chunk Large Files", 
            variable=self.auto_chunk_var
        ).pack(side=tk.LEFT, padx=5)
        
        # CUDA checkbox
        self.cuda_var = tk.BooleanVar(value=self.use_cuda)
        self.cuda_check = ttk.Checkbutton(
            settings_frame, 
            text="Use CUDA (GPU)", 
            variable=self.cuda_var,
            state=tk.NORMAL if self.use_cuda else tk.DISABLED
        )
        self.cuda_check.pack(anchor=tk.W)
        
        # Timestamp options
        self.timestamp_frame = ttk.Frame(settings_frame)
        self.timestamp_frame.pack(fill=tk.X, pady=5)
        
        self.timestamp_var = tk.BooleanVar(value=True)
        self.timestamp_check = ttk.Checkbutton(
            self.timestamp_frame, 
            text="Include Timestamps", 
            variable=self.timestamp_var,
            command=self.toggle_timestamp_options
        )
        self.timestamp_check.pack(side=tk.LEFT, padx=5)
        
        self.timestamp_level = tk.StringVar(value="word")
        self.word_radio = ttk.Radiobutton(
            self.timestamp_frame, 
            text="Word", 
            variable=self.timestamp_level, 
            value="word"
        )
        self.word_radio.pack(side=tk.LEFT, padx=5)
        
        self.segment_radio = ttk.Radiobutton(
            self.timestamp_frame, 
            text="Segment", 
            variable=self.timestamp_level, 
            value="segment"
        )
        self.segment_radio.pack(side=tk.LEFT, padx=5)
        
        # Output frame
        output_frame = ttk.LabelFrame(main_frame, text="Transcription", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, height=15)
        scrollbar = ttk.Scrollbar(output_frame, command=self.output_text.yview)
        self.output_text.config(yscrollcommand=scrollbar.set)
        
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            main_frame,
            orient=tk.HORIZONTAL,
            variable=self.progress_var,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.transcribe_btn = ttk.Button(
            button_frame, 
            text="Transcribe", 
            command=self.start_transcription_thread
        )
        self.transcribe_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Copy", command=self.copy_to_clipboard).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set(f"Ready. CUDA {'available' if self.use_cuda else 'not available'}")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, pady=5)

        # Initialize timestamp options state based on model
        self.update_timestamp_availability()
    
    def update_timestamp_availability(self):
        """Update timestamp availability based on selected model."""
        if self.model_var.get() == "ctc":
            # CTC doesn't support timestamps - disable timestamp options
            self.timestamp_var.set(False)
            self.timestamp_check.config(state=tk.DISABLED)
            self.word_radio.config(state=tk.DISABLED)
            self.segment_radio.config(state=tk.DISABLED)
        else:
            # TDT supports timestamps - enable timestamp options
            self.timestamp_check.config(state=tk.NORMAL)
            self.toggle_timestamp_options()

    def toggle_timestamp_options(self):
        """Enable/disable timestamp level options based on checkbox."""
        state = tk.NORMAL if self.timestamp_var.get() and self.model_var.get() != "ctc" else tk.DISABLED
        self.word_radio.config(state=state)
        self.segment_radio.config(state=state)
    
    def browse_file(self):
        """Select audio file."""
        if self.is_working:
            return
            
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.file_path.set(file_path)
    
    def clear_output(self):
        """Clear transcription output."""
        if not self.is_working:
            self.output_text.delete(1.0, tk.END)
    
    def copy_to_clipboard(self):
        """Copy transcription to clipboard."""
        if not self.is_working:
            text = self.output_text.get(1.0, tk.END).strip()
            if text:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.status_var.set("Copied to clipboard!")
    
    def save_transcription(self):
        """Save transcription to file."""
        if self.is_working:
            return
            
        text = self.output_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No transcription to save!")
            return
            
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("Success", f"Transcription saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def get_memory_usage(self):
        """Get current memory usage information with accurate VRAM reporting."""
        mem_info = {
            'ram_available': psutil.virtual_memory().available,
            'ram_total': psutil.virtual_memory().total,
            'ram_percent': psutil.virtual_memory().percent / 100.0,
            'vram_percent': 0.0  # Default value
        }
        
        # Add VRAM info if CUDA is available
        if self.use_cuda and torch.cuda.is_available():
            try:
                # Get VRAM stats - include both allocated and cached memory
                mem_info['vram_allocated'] = torch.cuda.memory_allocated(0)
                mem_info['vram_reserved'] = torch.cuda.memory_reserved(0)
                mem_info['vram_total'] = torch.cuda.get_device_properties(0).total_memory
                
                # Calculate percentage based on reserved memory (more accurate representation)
                # Reserved memory includes both allocated memory and cached memory
                if mem_info['vram_total'] > 0:
                    # Primary measurement: use reserved memory (includes both allocated and cached)
                    mem_info['vram_percent'] = mem_info['vram_reserved'] / mem_info['vram_total']
                    
                    # Add additional metrics for better debugging
                    mem_info['vram_allocated_percent'] = mem_info['vram_allocated'] / mem_info['vram_total']
                    
                    # Try to get more accurate info via nvidia-smi if available
                    try:
                        import subprocess
                        result = subprocess.check_output(
                            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
                            encoding='utf-8'
                        )
                        used_mem, total_mem = map(int, result.strip().split(','))
                        mem_info['vram_nvidia_used'] = used_mem * 1024 * 1024  # Convert MB to bytes
                        mem_info['vram_nvidia_total'] = total_mem * 1024 * 1024
                        mem_info['vram_nvidia_percent'] = used_mem / total_mem
                        
                        # Use nvidia-smi data as the primary source if available
                        mem_info['vram_percent'] = mem_info['vram_nvidia_percent']
                        print(f"NVIDIA-SMI reports: {used_mem}MB used / {total_mem}MB total = {mem_info['vram_percent']*100:.1f}%")
                    except Exception as e:
                        print(f"nvidia-smi not available, using PyTorch memory stats: {e}")
            except Exception as e:
                print(f"Error getting VRAM info: {e}")
                # If VRAM info cannot be retrieved, use system RAM instead
                mem_info['vram_percent'] = mem_info['ram_percent']
                
        return mem_info

    
    def estimate_chunk_size(self, audio_path):
        """Estimate appropriate chunk size based on audio length and available memory."""
        try:
            # Get audio info
            with sf.SoundFile(audio_path) as f:
                sample_rate = f.samplerate
                channels = f.channels
                duration = len(f) / f.samplerate  # in seconds
            
            # For very long files (>1 hour), be more aggressive with chunking
            if duration > 3600:  # More than 1 hour
                # Force a smaller chunk size for extremely long files
                return {
                    'duration': duration,
                    'chunk_size': min(self.default_chunk_duration, 1200),  # Max 20 minutes for long files
                    'num_chunks': int(np.ceil(duration / min(self.default_chunk_duration, 1200))),
                    'mem_info': self.get_memory_usage()
                }
            
            # Get current memory usage
            mem_info = self.get_memory_usage()
            
            # Calculate memory required for the entire file (rough estimate)
            # Each sample is a float32 (4 bytes)
            bytes_per_second = sample_rate * channels * 4
            total_bytes = bytes_per_second * duration
            
            # Determine limiting factor (RAM or VRAM)
            ram_available = (1 - self.memory_threshold) * mem_info['ram_total']
            
            if self.use_cuda and self.cuda_var.get():
                try:
                    vram_available = (1 - self.vram_threshold) * mem_info['vram_total']
                    limiting_memory = min(ram_available, vram_available)
                except:
                    limiting_memory = ram_available
            else:
                limiting_memory = ram_available
            
            # Calculate chunk size with a safety factor (0.5 instead of 0.8)
            chunk_size_seconds = max(30, min(duration, (limiting_memory / bytes_per_second) * 0.5))
            
            # Return results
            return {
                'duration': duration,
                'chunk_size': chunk_size_seconds,
                'num_chunks': int(np.ceil(duration / chunk_size_seconds)),
                'mem_info': mem_info
            }
            
        except Exception as e:
            print(f"Error estimating chunk size: {str(e)}")
            # Return a conservative default (2 minutes for safety)
            return {
                'duration': -1,  # Unknown
                'chunk_size': min(self.default_chunk_duration, 1200),  # Max 20 minutes
                'num_chunks': -1,  # Unknown
                'mem_info': self.get_memory_usage()
            }

    def transcribe_in_chunks(self, audio_path, chunk_size_seconds, include_timestamps, timestamp_level):
        """Transcribe audio file in chunks to manage memory."""
        try:
            # Get audio info
            with sf.SoundFile(audio_path) as f:
                sample_rate = f.samplerate
                total_frames = len(f)
                duration = total_frames / sample_rate
            
            # Calculate chunk size in frames
            chunk_size_frames = int(chunk_size_seconds * sample_rate)
            
            # Calculate number of chunks, with at least 10 chunks for very long files
            if duration > 3600:  # If longer than 1 hour
                # Ensure we have at least 10 chunks for very long files
                num_chunks = max(10, int(np.ceil(total_frames / chunk_size_frames)))
                chunk_size_frames = int(np.ceil(total_frames / num_chunks))
            else:
                num_chunks = int(np.ceil(total_frames / chunk_size_frames))
            
            # Initialize result container
            if include_timestamps:
                all_segments = []
                time_offset = 0.0
            else:
                all_text = []
            
            # Process each chunk
            for i in range(num_chunks):
                # Calculate chunk start and end
                start_frame = i * chunk_size_frames
                end_frame = min((i + 1) * chunk_size_frames, total_frames)
                chunk_duration = (end_frame - start_frame) / sample_rate
                
                # Update progress
                progress_percent = (i / num_chunks) * 100
                self.root.after(0, self.update_progress, progress_percent,
                    f"Processing chunk {i+1}/{num_chunks} ({progress_percent:.1f}%)"
                )
                
                # Extract chunk directly to temporary file to avoid keeping audio in memory
                temp_path = Path(f"temp_chunk_{i}.wav")
                
                # Process chunk with minimal memory usage
                with sf.SoundFile(audio_path) as input_file:
                    # Seek to start position
                    input_file.seek(start_frame)
                    
                    # Read chunk frames
                    chunk_frames = end_frame - start_frame
                    
                    # Create output file with appropriate parameters
                    output_sr = self.target_sample_rate
                    
                    with sf.SoundFile(temp_path, mode='w', samplerate=output_sr, 
                                    channels=1, format='WAV', subtype='PCM_16') as output_file:
                        
                        # Process in smaller blocks to minimize memory usage
                        # Use a max block size of 10 seconds of audio
                        block_size = min(int(10 * sample_rate), chunk_frames)
                        remaining_frames = chunk_frames
                        
                        while remaining_frames > 0:
                            # Read a block
                            current_block_size = min(block_size, remaining_frames)
                            audio_block = input_file.read(current_block_size, dtype='float32')
                            
                            # Convert stereo to mono if needed
                            if len(audio_block.shape) > 1:
                                audio_block = np.mean(audio_block, axis=1)
                            
                            # Resample block if needed
                            if sample_rate != output_sr:
                                audio_block = librosa.resample(
                                    audio_block, 
                                    orig_sr=sample_rate, 
                                    target_sr=output_sr
                                )
                            
                            # Write to output
                            output_file.write(audio_block)
                            
                            # Update remaining frames
                            remaining_frames -= current_block_size
                            
                            # Force cleanup
                            del audio_block
                            gc.collect()
                
                # Run transcription on the chunk
                if include_timestamps:
                    output = self.transcriber.transcribe_with_timestamps(temp_path)
                    
                    # Adjust timestamps based on chunk offset
                    if output and len(output) > 0 and 'timestamp' in output[0]:
                        for level in output[0]['timestamp']:
                            for item in output[0]['timestamp'][level]:
                                item['start'] += time_offset
                                item['end'] += time_offset
                        
                        all_segments.append(output)
                    
                    # Update time offset for next chunk
                    time_offset += chunk_duration
                else:
                    chunk_text = self.transcriber.transcribe_file(temp_path)
                    all_text.append(chunk_text)
                
                # Clean up temp file
                temp_path.unlink(missing_ok=True)
                
                # Force garbage collection after each chunk
                gc.collect()
                if self.use_cuda and self.cuda_var.get():
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
            
            # Combine results
            if include_timestamps:
                # Merge timestamped outputs
                combined_output = self.merge_timestamped_outputs(all_segments)
                return self.format_timestamped_output(combined_output, timestamp_level)
            else:
                # Combine plain text
                return " ".join(all_text)
                
        except Exception as e:
            raise Exception(f"Chunked transcription failed: {str(e)}")


    def start_transcription_thread(self):
        """Start transcription in a separate thread."""
        if self.is_working:
            return
            
        audio_path = self.file_path.get()
        if not audio_path:
            messagebox.showwarning("Warning", "Please select an audio file first!")
            return
            
        # Disable UI during processing
        self.is_working = True
        self.transcribe_btn.config(state=tk.DISABLED)
        self.status_var.set("Processing...")
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Analyzing file and preparing for transcription...\n")
        self.progress_var.set(0)
        
        # Get settings
        use_cuda = self.cuda_var.get() if self.use_cuda else False
        include_timestamps = self.timestamp_var.get()
        timestamp_level = self.timestamp_level.get() if include_timestamps else None
        auto_chunk = self.auto_chunk_var.get()
        
        # Start transcription thread
        threading.Thread(
            target=self.transcribe_audio,
            args=(audio_path, use_cuda, include_timestamps, timestamp_level, auto_chunk),
            daemon=True
        ).start()
    
    def transcribe_audio(self, audio_path, use_cuda, include_timestamps, timestamp_level, auto_chunk):
        """Main transcription function (runs in background thread)."""
        try:
            # Estimate chunk size for memory management
            chunk_info = self.estimate_chunk_size(audio_path)
            should_chunk = auto_chunk and chunk_info['num_chunks'] > 1
            
            # Update progress info
            memory_status = "RAM: {:.1f}%".format(chunk_info['mem_info']['ram_percent'] * 100)
            
            # Add VRAM info if using CUDA
            if use_cuda:
                vram_percent = chunk_info['mem_info'].get('vram_percent', 0) * 100
                memory_status += ", VRAM: {:.1f}%".format(vram_percent)
                
                # Debug output to console
                print(f"VRAM info: {chunk_info['mem_info'].get('vram_allocated', 0) / 1024**2:.2f} MB allocated, "
                    f"{chunk_info['mem_info'].get('vram_total', 0) / 1024**2:.2f} MB total, "
                    f"{vram_percent:.2f}% used")
            
            # Update status with chunking info
            if should_chunk:
                self.root.after(0, self.update_status,
                    f"File duration: {chunk_info['duration']:.1f}s. "
                    f"Processing in {chunk_info['num_chunks']} chunks of {chunk_info['chunk_size']:.1f}s each. "
                    f"{memory_status}"
                )
            else:
                self.root.after(0, self.update_status,
                    f"File duration: {chunk_info['duration']:.1f}s. "
                    f"Processing as a single file. {memory_status}"
                )
            
            # Initialize transcriber
            if self.transcriber is None or self.transcriber.use_cuda != use_cuda:
                if self.model_var.get() == "tdt":
                    self.transcriber = TDTTranscriber(use_cuda=use_cuda)
                else:
                    self.transcriber = CTCTranscriber(use_cuda=use_cuda)
            
            if should_chunk:
                transcription = self.transcribe_in_chunks(
                    audio_path, 
                    chunk_info['chunk_size'], 
                    include_timestamps, 
                    timestamp_level
                )
            else:
                # Process entire file
                transcription = self.transcribe_whole_file(
                    audio_path, 
                    include_timestamps, 
                    timestamp_level
                )
            
            # Update GUI with final result
            self.root.after(0, self.show_transcription_result, transcription, None)
            
        except Exception as e:
            self.root.after(0, self.show_transcription_result, None, str(e))
    
    def update_status(self, message):
        """Update status bar message (thread-safe)."""
        self.status_var.set(message)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, message + "\n\nTranscribing... Please wait...")
    
    def update_progress(self, value, progress_message):
        """Update progress bar and message (thread-safe)."""
        self.progress_var.set(value)
        self.status_var.set(progress_message)
        
        # Update progress in text area as well
        text_content = self.output_text.get(1.0, tk.END)
        lines = text_content.split('\n')
        if len(lines) > 3:
            new_content = '\n'.join(lines[:-2]) + f"\n\n{progress_message}"
        else:
            new_content = progress_message
        
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, new_content)
    
    def transcribe_whole_file(self, audio_path, include_timestamps, timestamp_level):
        """Transcribe the entire audio file at once."""
        # Load and prepare audio
        audio, sr = self.load_audio(audio_path)
        
        # Save to temp file for transcriber
        temp_path = Path("temp_16k.wav")
        sf.write(temp_path, audio, self.target_sample_rate)
        
        # Clear memory
        del audio
        gc.collect()
        
        # Perform transcription
        if include_timestamps:
            output = self.transcriber.transcribe_with_timestamps(temp_path)
            transcription = self.format_timestamped_output(output, timestamp_level)
        else:
            transcription = self.transcriber.transcribe_file(temp_path)
        
        # Clean up
        temp_path.unlink(missing_ok=True)
        return transcription
    
    
    def merge_timestamped_outputs(self, outputs):
        """Merge multiple timestamped outputs into a single output."""
        if not outputs:
            return None
            
        # Initialize merged output with structure from first output
        merged = [{'text': '', 'timestamp': {}}]
        
        # Combine text
        all_text = []
        for output in outputs:
            if output and output[0].get('text'):
                all_text.append(output[0]['text'])
        
        merged[0]['text'] = ' '.join(all_text)
        
        # Merge timestamps for each level (word, segment, char)
        timestamp_levels = set()
        for output in outputs:
            if output and output[0].get('timestamp'):
                timestamp_levels.update(output[0]['timestamp'].keys())
        
        for level in timestamp_levels:
            merged[0]['timestamp'][level] = []
            
            # Collect all timestamps for this level
            for output in outputs:
                if output and output[0].get('timestamp') and output[0]['timestamp'].get(level):
                    merged[0]['timestamp'][level].extend(output[0]['timestamp'][level])
            
            # Sort by start time
            merged[0]['timestamp'][level].sort(key=lambda x: x.get('start', 0))
        
        return merged
    
    def load_audio(self, audio_path):
        """Load and preprocess audio file."""
        try:
            audio, sr = sf.read(audio_path, dtype='float32')
            if len(audio.shape) > 1:  # Convert stereo to mono
                audio = np.mean(audio, axis=1)
            if sr != self.target_sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)
            return audio, sr
        except Exception as e:
            raise Exception(f"Failed to load audio: {str(e)}")
    
    def format_timestamped_output(self, output, level="word"):
        """Format timestamped output for display."""
        if not output or not isinstance(output, list) or not output[0].get('timestamp'):
            return "No timestamp data available"
            
        timestamps = output[0]['timestamp'].get(level, [])
        if not timestamps:
            return f"No {level}-level timestamps available"
            
        result = []
        for stamp in timestamps:
            if level == "segment":
                text = stamp.get('segment', '')
            elif level == "word":
                text = stamp.get('word', '')
            else:  # char
                text = stamp.get('char', '')
                
            start = stamp.get('start', 0)
            end = stamp.get('end', 0)
            result.append(f"[{start:.2f}s - {end:.2f}s] {text}")
            
        return "\n".join(result)
    
    def show_transcription_result(self, transcription, error):
        """Show transcription result or error in GUI."""
        self.is_working = False
        self.transcribe_btn.config(state=tk.NORMAL)
        
        if error:
            self.status_var.set(f"Error: {error}")
            self.output_text.delete(1.0, tk.END)
            messagebox.showerror("Error", f"Transcription failed: {error}")
        else:
            self.status_var.set("Transcription complete")
            self.output_text.delete(1.0, tk.END)
            
            # Check if we should format as SRT
            if self.output_format.get() == "srt" and isinstance(transcription, list) and transcription and "timestamp" in transcription[0]:
                # If transcription is already in timestamped format
                formatted = self.format_srt_output(transcription)
                self.output_text.insert(tk.END, formatted)
            elif self.output_format.get() == "srt" and self.timestamp_var.get():
                # For text with timestamps that needs to be converted to SRT
                lines = transcription.strip().split("\n")
                srt_content = self.convert_timestamped_text_to_srt(lines)
                self.output_text.insert(tk.END, srt_content)
            else:
                # Regular text output
                self.output_text.insert(tk.END, transcription)

    def convert_timestamped_text_to_srt(self, lines):
        """Convert timestamped text format to SRT format."""
        srt_lines = []
        counter = 1
        
        for line in lines:
            # Parse timestamp and text from format like "[0.50s - 1.20s] word"
            if not line.startswith("[") or "]" not in line:
                continue
                
            timestamp_part = line[1:line.find("]")]
            text_part = line[line.find("]")+1:].strip()
            
            # Parse start and end times
            if " - " in timestamp_part:
                start_str, end_str = timestamp_part.split(" - ")
                start = float(start_str.replace("s", ""))
                end = float(end_str.replace("s", ""))
                
                # Format as SRT entry
                srt_lines.append(
                    f"{counter}\n"
                    f"{self._seconds_to_srt_time(start)} --> {self._seconds_to_srt_time(end)}\n"
                    f"{text_part}\n"
                )
                counter += 1
        
        return "\n".join(srt_lines)

    def format_srt_output(self, output):
        """Properly format timestamped output as SRT subtitles."""
        level = self.timestamp_level.get()
        
        if not output or not isinstance(output, list) or not output[0].get('timestamp'):
            return "No timestamp data available"
        
        timestamps = output[0]['timestamp'].get(level, [])
        if not timestamps:
            return f"No {level}-level timestamps available"
        
        srt_lines = []
        counter = 1
        
        for stamp in timestamps:
            # Get text based on level
            if level == "segment":
                text = stamp.get('segment', '').replace("▁", " ").strip()
            elif level == "word":
                text = stamp.get('word', '').replace("▁", " ").strip()
            else:  # char
                text = stamp.get('char', '')
            
            # Skip empty segments
            if not text:
                continue
                
            # Get timestamps (ensure end > start)
            start = max(0, stamp.get('start', 0))
            end = max(start + 0.001, stamp.get('end', start + 1))  # Ensure minimum duration
            
            # Format as SRT
            srt_lines.append(
                f"{counter}\n"
                f"{self._seconds_to_srt_time(start)} --> {self._seconds_to_srt_time(end)}\n"
                f"{text}\n"
            )
            counter += 1
        
        return "\n".join(srt_lines)

    def _seconds_to_srt_time(self, seconds):
        """Convert seconds to proper SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remaining = seconds % 60
        milliseconds = int((seconds_remaining - int(seconds_remaining)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds_remaining):02d},{milliseconds:03d}"

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Audio Transcription Tool")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--model", choices=["tdt", "ctc"], default="tdt", help="Model type")
    parser.add_argument("--input", help="Input audio file")
    parser.add_argument("--output", help="Output text file")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps")
    parser.add_argument("--level", choices=["word", "segment", "char"], default="word", help="Timestamp level")
    parser.add_argument("--format", choices=["text", "srt"], default="text", help="Output format (text or srt)")
    
    args = parser.parse_args()
    
    if args.gui or not args.input:
        root = tk.Tk()
        app = AudioTranscriberApp(root)
        root.mainloop()
    else:
        try:
            # Check CUDA availability
            use_cuda = args.cuda and "CUDAExecutionProvider" in ort.get_available_providers()
            
            # Initialize transcriber
            if args.model == "tdt":
                transcriber = TDTTranscriber(use_cuda=use_cuda)
            else:
                transcriber = CTCTranscriber(use_cuda=use_cuda)

            # Reject timestamp requests for CTC model
            if args.timestamps and args.model == "ctc":
                print("Warning: Timestamps are not supported for CTC model. Proceeding without timestamps.")
                args.timestamps = False
            
            # Transcribe
            if args.timestamps and args.model == "tdt":
                output = transcriber.transcribe_with_timestamps(Path(args.input))
                
                if args.format == "srt":
                    # Create proper SRT with sentence preservation
                    app = AudioTranscriberApp(None)  # Create dummy app to use its methods
                    transcription = app.format_srt_output(output, args.level)
                else:
                    # Create text with timestamps
                    transcription = "\n".join(
                        f"[{stamp['start']:.2f}s - {stamp['end']:.2f}s] {stamp[args.level]}"
                        for stamp in output[0]['timestamp'][args.level]
                    )
            else:
                transcription = transcriber.transcribe_file(Path(args.input))
                if args.format == "srt":
                    # Create estimated SRT from plain text
                    app = AudioTranscriberApp(None)  # Create dummy app to use its methods
                    transcription = app.text_to_srt(transcription)
            
            # Save or print result
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                print(f"Saved to {args.output}")
            else:
                print(transcription)
                
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
