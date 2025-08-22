#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 17:14:40 2025

@author: umbertocappellazzo
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class AudioVisualTokenBenchmark:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B"):
        """
        Initialize the benchmark with Llama 3.1-8B model
        """
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on device: {self.device}")

    def create_mock_audiovisual_tokens(self, audio_tokens: int, video_tokens: int) -> torch.Tensor:
        """
        Create mock audio-visual tokens by generating random token IDs
        In practice, these would come from audio/video encoders
        """
        # Create random token IDs within vocabulary range
        vocab_size = len(self.tokenizer)
        
        # Generate audio tokens
        audio_token_ids = torch.randint(0, vocab_size, (audio_tokens,))
        
        # Generate video tokens  
        video_token_ids = torch.randint(0, vocab_size, (video_tokens,))
        
        # Concatenate audio and video tokens
        av_tokens = torch.cat([audio_token_ids, video_token_ids])
        
        return av_tokens.to(self.device)

    def apply_compression(self, tokens: torch.Tensor, compression_rate: int) -> torch.Tensor:
        """
        Apply compression by taking every nth token (simple downsampling)
        In practice, this could be more sophisticated compression like PCA, clustering, etc.
        """
        if compression_rate <= 1:
            return tokens
        
        # Simple downsampling - take every nth token
        compressed_tokens = tokens[::compression_rate]
        return compressed_tokens

    def create_prompt_with_av_tokens(self, av_tokens: torch.Tensor, prompt_text: str = "Describe what you observe:") -> torch.Tensor:
        """
        Create input sequence with audio-visual tokens followed by text prompt
        """
        # Tokenize the text prompt
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        prompt_tokens = prompt_tokens.squeeze(0).to(self.device)
        
        # Concatenate AV tokens with prompt tokens
        full_input = torch.cat([av_tokens, prompt_tokens])
        
        return full_input.unsqueeze(0)  # Add batch dimension

    def benchmark_generation(self, 
                           audio_tokens: int, 
                           video_tokens: int, 
                           compression_rates: List[int],
                           prompt_text: str = "Describe what you observe:",
                           max_new_tokens: int = 30,
                           num_runs: int = 3) -> Dict:
        """
        Benchmark generation performance across different compression rates
        """
        results = []
        
        print(f"\nBenchmarking with {audio_tokens} audio tokens and {video_tokens} video tokens")
        print(f"Target response length: {max_new_tokens} tokens")
        print("-" * 60)
        
        for compression_rate in compression_rates:
            print(f"\nTesting compression rate: {compression_rate}")
            
            # Create mock audio-visual tokens
            av_tokens = self.create_mock_audiovisual_tokens(audio_tokens, video_tokens)
            
            # Apply compression
            compressed_av_tokens = self.apply_compression(av_tokens, compression_rate)
            
            # Create full input with prompt
            input_ids = self.create_prompt_with_av_tokens(compressed_av_tokens, prompt_text)
            
            # Calculate token counts
            original_av_count = len(av_tokens)
            compressed_av_count = len(compressed_av_tokens) 
            prompt_token_count = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))
            total_input_tokens = compressed_av_count + prompt_token_count
            
            # Run multiple times for average
            generation_times = []
            tokens_per_second_list = []
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}... ", end="")
                
                # Measure generation time
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Deterministic for consistent benchmarking
                        pad_token_id=self.tokenizer.eos_token_id,
                        attention_mask=torch.ones_like(input_ids)
                    )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Calculate actual tokens generated
                generated_tokens = outputs.shape[1] - input_ids.shape[1]
                tokens_per_second = generated_tokens / generation_time
                
                generation_times.append(generation_time)
                tokens_per_second_list.append(tokens_per_second)
                
                print(f"{generation_time:.3f}s, {tokens_per_second:.1f} tok/s")
            
            # Calculate averages
            avg_generation_time = np.mean(generation_times)
            avg_tokens_per_second = np.mean(tokens_per_second_list)
            std_generation_time = np.std(generation_times)
            std_tokens_per_second = np.std(tokens_per_second_list)
            
            # Store results
            result = {
                'compression_rate': compression_rate,
                'original_audio_tokens': audio_tokens,
                'original_video_tokens': video_tokens,
                'original_av_tokens': original_av_count,
                'compressed_av_tokens': compressed_av_count,
                'prompt_tokens': prompt_token_count,
                'total_input_tokens': total_input_tokens,
                'target_response_tokens': max_new_tokens,
                'avg_generation_time': avg_generation_time,
                'std_generation_time': std_generation_time,
                'avg_tokens_per_second': avg_tokens_per_second,
                'std_tokens_per_second': std_tokens_per_second,
                'compression_ratio': original_av_count / compressed_av_count if compressed_av_count > 0 else float('inf')
            }
            
            results.append(result)
            
            print(f"  Average: {avg_generation_time:.3f}±{std_generation_time:.3f}s, "
                  f"{avg_tokens_per_second:.1f}±{std_tokens_per_second:.1f} tok/s")
            print(f"  AV tokens: {original_av_count} → {compressed_av_count} "
                  f"({compressed_av_count/original_av_count*100:.1f}% retained)")


    def create_results_dataframe(self, results: List[Dict]) -> pd.DataFrame:
            """
            Convert results to pandas DataFrame for easy analysis
            """
            df = pd.DataFrame(results)
            return df

    def plot_results(self, df: pd.DataFrame, save_path: str = None):
        """
        Create visualization of benchmark results
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Generation time vs compression rate
        ax1.errorbar(df['compression_rate'], df['avg_generation_time'], 
                    yerr=df['std_generation_time'], marker='o', capsize=5)
        ax1.set_xlabel('Compression Rate')
        ax1.set_ylabel('Generation Time (seconds)')
        ax1.set_title('Generation Time vs Compression Rate')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Tokens per second vs compression rate
        ax2.errorbar(df['compression_rate'], df['avg_tokens_per_second'], 
                    yerr=df['std_tokens_per_second'], marker='s', capsize=5, color='orange')
        ax2.set_xlabel('Compression Rate')
        ax2.set_ylabel('Tokens per Second')
        ax2.set_title('Generation Speed vs Compression Rate')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Total input tokens vs compression rate
        ax3.plot(df['compression_rate'], df['total_input_tokens'], marker='^', color='green')
        ax3.set_xlabel('Compression Rate')
        ax3.set_ylabel('Total Input Tokens')
        ax3.set_title('Input Size vs Compression Rate')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Efficiency (tokens/s per input token)
        efficiency = df['avg_tokens_per_second'] / df['total_input_tokens']
        ax4.plot(df['compression_rate'], efficiency, marker='d', color='red')
        ax4.set_xlabel('Compression Rate')
        ax4.set_ylabel('Generation Efficiency (tok/s per input token)')
        ax4.set_title('Generation Efficiency vs Compression Rate')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def print_summary_table(self, df: pd.DataFrame):
        """
        Print a formatted summary table
        """
        print("\n" + "="*100)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*100)
        
        print(f"{'Compression':<12} {'AV Tokens':<12} {'Input Tokens':<13} {'Gen Time':<12} {'Speed':<15} {'Efficiency':<12}")
        print(f"{'Rate':<12} {'(Compressed)':<12} {'(Total)':<13} {'(seconds)':<12} {'(tok/s)':<15} {'(tok/s/input)':<12}")
        print("-"*100)
        
        for _, row in df.iterrows():
            efficiency = row['avg_tokens_per_second'] / row['total_input_tokens']
            print(f"{row['compression_rate']:<12} "
                  f"{row['compressed_av_tokens']:<12} "
                  f"{row['total_input_tokens']:<13} "
                  f"{row['avg_generation_time']:<12.3f} "
                  f"{row['avg_tokens_per_second']:<15.1f} "
                  f"{efficiency:<12.4f}")

# Example usage and benchmark execution
def run_benchmark_example():
    """
    Run the benchmark with your specified parameters
    """
    # Initialize benchmark
    benchmark = AudioVisualTokenBenchmark()
    
    # Your specified parameters
    audio_tokens = 1000
    video_tokens = 500
    prompt_text = "Describe what you observe:"  # This will be ~5 tokens
    max_new_tokens = 30  # Target response length
    
    # Compression rates to test (1 means no compression)
    compression_rates = [1, 4, 8, 20]
    
    # Run benchmark
    results = benchmark.benchmark_generation(
        audio_tokens=audio_tokens,
        video_tokens=video_tokens,
        compression_rates=compression_rates,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
        num_runs=3  # Average over 3 runs for more reliable results
    )
    
    # Convert to DataFrame
    df = benchmark.create_results_dataframe(results)
    
    # Print summary
    benchmark.print_summary_table(df)
    
    # Create plots
    #benchmark.plot_results(df, save_path='llama_av_benchmark.png')
    
    # Additional analysis
    print(f"\nKEY INSIGHTS:")
    print(f"- Original A/V tokens: {audio_tokens + video_tokens}")
    print(f"- Compression rates tested: {compression_rates}")
    print(f"- Best speed: {df['avg_tokens_per_second'].max():.1f} tok/s "
          f"(compression rate {df.loc[df['avg_tokens_per_second'].idxmax(), 'compression_rate']})")
    print(f"- Speed improvement with max compression: "
          f"{df['avg_tokens_per_second'].iloc[-1] / df['avg_tokens_per_second'].iloc[0]:.2f}x")
    
    return df

# Run the benchmark
if __name__ == "__main__":
    # Make sure you have the required packages installed:
    # pip install torch transformers accelerate matplotlib pandas numpy
    
    # Note: You'll need to have access to Llama 3.1-8B model
    # Either through Hugging Face Hub (requires authentication) or local model files
    
    results_df = run_benchmark_example()