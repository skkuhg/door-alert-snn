# Door Alert SNN System

A spiking neural network (SNN) implementation for door timeout alerts using Leaky Integrate-and-Fire (LIF) neurons.

## Overview

This project implements a door monitoring system that uses a LIF neuron to detect when a door has been left open for too long. The system generates alerts after a configurable timeout period and includes a cooldown mechanism to prevent alert spam.

## Features

- **LIF Neuron Timer**: Uses a Leaky Integrate-and-Fire neuron model as a biological timer
- **Configurable Timeout**: Set custom timeout thresholds (default: 20 seconds)
- **Cooldown Period**: Prevents alert spam with configurable cooldown (default: 10 seconds)
- **Multiple Scenarios**: Supports both structured and stochastic door behavior patterns
- **Example Data Generation**: Generate synthetic test data for various scenarios

## Files

- `generate_example_data.py`: Main script for generating synthetic door behavior data
- `example_test_data.py`: Pre-configured test scenarios for quick testing
- `example_data/`: Directory containing generated CSV files and metadata

## Usage

### Generate Example Data

```bash
python generate_example_data.py --scenario mixed --sim-minutes 30
```

Options:
- `--scenario`: Choose 'mixed' (structured events) or 'stochastic' (random events)
- `--dt`: Time step in seconds (default: 0.1)
- `--sim-minutes`: Simulation duration in minutes (default: 30)
- `--outdir`: Output directory (default: 'example_data')
- `--seed`: Random seed for reproducibility (default: 42)

### Generated Files

The script generates:
- `door_events.csv`: Time series of door states with segment IDs
- `simulation_output.csv`: Full simulation data including neuron voltage and alerts
- `timeouts.csv`: Ground truth timeout events
- `metadata.json`: Simulation parameters and statistics

## Parameters

- **Open Timeout**: 20 seconds (when alerts should trigger)
- **Cooldown Period**: 10 seconds (minimum time between alerts)
- **Membrane Time Constant**: 10 seconds
- **Threshold Voltage**: 1.0
- **Reset Voltage**: 0.0

## Scenarios

### Mixed Scenario
- Short opens (3-5s): No alerts
- Medium opens (15s): No alerts  
- Long opens (25-45s): Alerts triggered
- Rapid open-close patterns: Tests system robustness

### Stochastic Scenario
- Random door behavior with occasional long opens
- Tests system performance on realistic usage patterns

## Requirements

- numpy
- matplotlib (for visualization in notebooks)
- pathlib
- csv
- json