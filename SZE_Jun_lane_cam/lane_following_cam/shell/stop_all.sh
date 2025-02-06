#!/bin/bash

# STOP

echo "[INFO] Killing Screens"
killall -9 screen
echo "[INFO] Wiping empty Screens"
screen -wipe