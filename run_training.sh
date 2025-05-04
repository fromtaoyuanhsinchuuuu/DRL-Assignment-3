#!/bin/bash
# run_training_fixed.sh - Script to restart training from a checkpoint with reset epsilon

# Default values
EPISODE=2000
EPSILON=0.2

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --episode)
      EPISODE="$2"
      shift 2
      ;;
    --epsilon)
      EPSILON="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Restarting training from episode $EPISODE with epsilon $EPSILON"

# First, reset the epsilon in the checkpoint
python3.8 restart_training.py --episode $EPISODE --epsilon $EPSILON

# Then start the training
if [ $? -eq 0 ]; then
  echo "Successfully reset epsilon. Starting training..."
  python3.8 train_dueling_nstep.py
else
  echo "Failed to reset epsilon. Please check the error messages above."
fi
