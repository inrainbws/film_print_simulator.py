#!/usr/bin/env python3
# Copyright (c) 2025 @inrainbws (Github)
# Film Print Simulator - A tool for simulating analog prints from film scans
# https://github.com/inrainbws/film_print_simulator.py

import numpy as np
import argparse
from pathlib import Path
import tifffile as tiff
import cv2  # For saving JPG files
import time  # For performance timing
import rawpy  # For reading RAW files
import os  # For file extension handling

# Global debug flag
DEBUG = False


def calculate_film_density(image):
    """
    Convert input image (transmittance) to density
    d_i = log10(I_0,i / I_i) = -log10(I_i / I_0,i) = -log10(T_i)
    
    Assumes image values are transmittance in range [0, 1]
    """
    # Avoid log(0) by clipping small values
    epsilon = 1e-12
    image_clipped = np.maximum(image, epsilon)
    return -np.log10(image_clipped)


def normalize_brightness(brightness_values):
    """
    Args:
        brightness_values: List or array of brightness values for the CMY color heads

    Returns:
        Normalized array of brightness values
    """
    if not isinstance(brightness_values, np.ndarray):
        brightness_values = np.array(brightness_values)

    return brightness_values / np.mean(brightness_values)


def process_channel(i, density_channel, brightness_i, D_min, gamma):

    density_channel = density_channel - D_min + np.log10(brightness_i)
    reflectance = np.power(10., gamma * density_channel)

    return reflectance

def exponential_model(density_image, brightness, film_params, gamma=2.0):

    # Number of channels
    num_channels = density_image.shape[2] if len(density_image.shape) > 2 else 1
    
    # Initialize the result array
    result = np.zeros_like(density_image)
    
    # Assign results back to the array
    for i in range(num_channels):
        result[..., i] = process_channel(i, density_image[..., i], brightness[i], film_params["D_min"][i], gamma)
        
    return result

def apply_gamma_correction(image, gamma=2.2):
    """
    Apply gamma correction to an image for better screen display
    
    Args:
        image: Input image with values in range [0, 1]
        gamma: Gamma value (typically 2.2 for sRGB display)
        
    Returns:
        Gamma-corrected image with values in range [0, 1]
    """
    return np.power(image, 1/gamma)

def apply_srgb_transfer_function(image):
    """
    Apply the standard sRGB transfer function to linear RGB values
    
    Args:
        image: Input image with linear RGB values in range [0, 1]
        
    Returns:
        Image with sRGB transfer function applied, values in range [0, 1]
    """
    # Create output array with same shape as input
    result = np.zeros_like(image)
    
    # Apply the piecewise sRGB transfer function
    # For values <= 0.0031308: 12.92 * linear
    # For values > 0.0031308: 1.055 * linear^(1/2.4) - 0.055
    mask = image <= 0.0031308
    result[mask] = 12.92 * image[mask]
    result[~mask] = 1.055 * np.power(image[~mask], 1/2.4) - 0.055
    
    return result

def simulate_print(input_image, brightness_cmy, alpha=4.0, overall_brightness=1.0):
    """
    Simulate the analog print process using sigmoidal model
    
    Args:
        input_image: Input film scan (transmittance values in range [0, 1])
        brightness_cmy: List of brightness values for CMY color heads
        alpha: Alpha value for sigmoidal model
        overall_brightness: Overall brightness multiplier
    
    Returns:
        Simulated print (reflectance values in range [0, 1])
    """
    start_time = time.time()
    
    # Ensure input is float and in range [0, 1]
    input_float = input_image.astype(np.float32)
    
    # If input has 4 channels (RGBA), just use first 3
    if input_float.shape[2] == 4:
        input_float = input_float[..., :3]
    
    # Calculate film density
    density_image = calculate_film_density(input_float)
    
    # Normalize brightness values
    brightness_normalized = normalize_brightness(brightness_cmy)
    
    # Apply overall brightness multiplier
    brightness_normalized = brightness_normalized * overall_brightness

    film_params = {"D_min": np.array([0.27, 0.49, 1.13])}
    print_image = exponential_model(density_image, brightness_normalized, film_params, alpha)

    # Clip and normalize output
    output_image = np.clip(print_image, 0, 1)
    
    end_time = time.time()
    if DEBUG:
        print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    return output_image

def main():
    parser = argparse.ArgumentParser(description='Simulate analog print from color negative film')
    parser.add_argument('input_file', type=str, help='Input file (supported formats: TIFF, RAW, JPEG)')
    parser.add_argument('output_file', type=str, help='Output 32-bit float TIFF file (simulated print)')
    parser.add_argument('--cyan', type=float, default=1.0, help='Brightness of cyan color head')
    parser.add_argument('--magenta', type=float, default=1.0, help='Brightness of magenta color head')
    parser.add_argument('--yellow', type=float, default=1.0, help='Brightness of yellow color head')
    parser.add_argument('--overall-brightness', type=float, default=1.0, help='Overall brightness multiplier')
    parser.add_argument('--alpha', type=float, default=4.0, help='Alpha value for sigmoidal model')
    parser.add_argument('--display-gamma', type=float, default=2.2, 
                        help='Gamma value for preview JPG (default: 2.2 for sRGB displays)')
    parser.add_argument('--no-jpg', action='store_true', help='Skip generating preview JPG file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Set global debug flag
    global DEBUG
    DEBUG = args.debug
    
    # Start timing
    start_time = time.time()
    
    # Get file extension to determine file type
    file_ext = os.path.splitext(args.input_file)[1].lower()
    
    # Read input file based on file type
    try:
        # List of supported RAW file extensions
        raw_extensions = ['.dng', '.nef', '.raf', '.arw', '.cr2', '.cr3', '.orf', '.rw2', '.pef', '.srw']
        
        if file_ext in raw_extensions:
            # Load RAW file using rawpy
            with rawpy.imread(args.input_file) as raw:
                # Get the linear RGB image
                input_image = raw.postprocess(
                    gamma=(1, 1),  # Linear output
                    no_auto_bright=True,
                    output_bps=16,
                    user_wb=[1, 1, 1, 1]  # No white balance adjustment
                ).astype(np.float32) / 65535.0  # Normalize to [0, 1]
        
        elif file_ext in ['.tif', '.tiff']:
            # Load TIFF file
            input_image = tiff.imread(args.input_file).astype(np.float32)
            
            # Normalize if necessary
            if input_image.max() > 1.0:
                input_image = input_image / input_image.max()
        
        elif file_ext in ['.jpg', '.jpeg']:
            # Load JPEG (not recommended for film scans, but supported)
            img = cv2.imread(args.input_file)
            if img is None:
                raise ValueError("Could not read image file")
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            input_image = img.astype(np.float32) / 255.0
        
        else:
            # Fall back to trying tiff.imread for unknown formats
            input_image = tiff.imread(args.input_file).astype(np.float32)
            if input_image.max() > 1.0:
                input_image = input_image / input_image.max()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Ensure we have a 3-channel RGB image
    if len(input_image.shape) == 2:
        # Convert grayscale to RGB
        input_image = np.stack([input_image] * 3, axis=-1)
    elif input_image.shape[2] > 3:
        # Use only the first 3 channels
        input_image = input_image[:, :, :3]
    
    # Brightness values for CMY color heads
    brightness_cmy = np.array([args.cyan, args.magenta, args.yellow])
    
    # Simulate print
    output_image = simulate_print(
        input_image, 
        brightness_cmy,
        alpha=args.alpha,
        overall_brightness=args.overall_brightness,
    )
    
    # Save 32-bit TIFF output
    try:
        output_image = (output_image * 65535.0).astype(np.uint16)
        tiff.imwrite(args.output_file, output_image)
        print(f"Output saved to {args.output_file}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        
    # Save preview JPG with gamma correction
    if not args.no_jpg:
        try:
            # Generate JPG filename by replacing extension or adding .jpg
            output_path = Path(args.output_file)
            jpg_path = output_path.with_suffix('.jpg')
            
            # Apply sRGB transfer function for display
            display_image = apply_srgb_transfer_function(output_image)
            
            # Convert to 8-bit for JPG
            jpg_image = (display_image * 255).astype(np.uint8)
            
            # OpenCV expects BGR order
            if jpg_image.shape[2] == 3:  # Check if we have 3 color channels
                jpg_image = cv2.cvtColor(jpg_image, cv2.COLOR_RGB2BGR)
            
            # Save JPG file
            cv2.imwrite(str(jpg_path), jpg_image)
            print(f"Preview JPG saved to {jpg_path}")
        except Exception as e:
            print(f"Error saving JPG preview: {e}")
    
    # End timing
    end_time = time.time()
    if DEBUG:
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 